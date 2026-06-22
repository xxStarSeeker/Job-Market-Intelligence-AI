"""
Extract a clean technical skill vocabulary from the 11 tech-role job
descriptions with Ollama (qwen2.5:32b) -> scripts/skill_vocab.py.

Fixes the old version's empty arrays via (1) `focus_text()` front-loading the
Requirements/Responsibilities section past the "About us" fluff, (2) a few-shot
prompt demanding concrete tools/tech/certs, and (3) a cleanup pass that applies
ALIASES/SYNONYMS and drops company names, languages, soft skills, platforms.
Resumable (JSONL), thread-safe (cf. label_jobs.py).

    python scripts/build_skill_vocab_llm.py --workers 4            # full run
    python scripts/build_skill_vocab_llm.py --aggregate-only --min-freq 5
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL",
                                 os.environ.get("OLLAMA_URL", "http://localhost:11434"))
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")
OLLAMA_API_TAGS = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/tags")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct-q4_K_M")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
JOBS_RAW_CSV = PROJECT_ROOT / "job_data_saudi_clean.csv"
OUTPUT_VOCAB = PROJECT_ROOT / "scripts" / "skill_vocab.py"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "extracted_skills.jsonl"

MIN_SKILL_FREQ = 5
MAX_FOCUS_CHARS = 9000      # ~2700 tokens; fits comfortably in num_ctx 8192

TECH_CLASSES = [
    "Data Engineer", "Business Analyst", "Data Analyst",
    "Cybersecurity Analyst", "AI Engineer", "Cloud Engineer",
    "Data Scientist", "Software Engineer", "DevOps Engineer",
    "Product Manager", "QA Engineer",
]

GOLD_HEADER_RE = re.compile(    # headers introducing the skills/tools section
    r"(key responsibilities|responsibilities|requirements|qualifications|"
    r"required skills|technical skills|skills (?:and|&) |what you['’]?ll do|"
    r"what we['’]?re looking for|who you are|the role|your profile|must have|"
    r"minimum qualifications|preferred qualifications|job duties|"
    r"experience required|desired skills)", re.IGNORECASE)

SKILL_PROMPT = """You extract TECHNICAL skills from job descriptions for a skills database.
Return a JSON object {{"skills": [ ... ]}} listing every concrete technical tool, \
technology, framework, library, platform, programming/query language, database, \
cloud service, certification, or engineering methodology named anywhere in the \
text. Skills are often buried under headings like "Requirements" — scan it ALL.
INCLUDE tools/tech (Python, SQL, Apache Spark, Docker, Kubernetes, Power BI, \
TensorFlow, React), methodologies (CI/CD, ETL, Agile, MLOps), and certs (CISSP, \
PMP, CCNA). EXCLUDE company names, locations, spoken languages (English/Arabic), \
soft skills (communication, teamwork), job titles, degrees, and social platforms \
(LinkedIn). Use canonical short names (1-4 words).
Example input: "About us: a leading firm. Requirements: 3+ years Python and SQL. \
Build ETL pipelines on AWS (S3, Redshift) with Apache Airflow. Strong \
communication. Fluent English. Docker a plus."
Example output: {{"skills": ["Python", "SQL", "ETL", "AWS", "S3", "Redshift", \
"Apache Airflow", "Docker"]}}
Return ONLY the JSON object for this job description.

Job description:
{description}"""

SPOKEN_LANGUAGES = set("""english arabic french spanish german hindi urdu
mandarin chinese russian portuguese italian japanese korean turkish bilingual""".split())
PLATFORMS_BLOCK = set("""linkedin facebook twitter instagram youtube tiktok x
whatsapp telegram snapchat indeed glassdoor""".split())
GENERIC_BLOCK = set("""experience knowledge skills skill ability degree bachelor
bachelors master masters phd diploma certification certifications tools systems
technology technologies software hardware framework frameworks platform platforms
database databases programming coding it data cloud web api apis etc""".split()) | {
    "computer science", "information technology", "best practices"}
SOFT_SKILLS = set("""communication teamwork leadership collaboration adaptability
creativity negotiation flexibility""".split()) | {
    "team work", "problem solving", "problem-solving", "time management",
    "critical thinking", "interpersonal skills", "attention to detail",
    "analytical skills", "presentation skills", "decision making", "soft skills"}
ALIASES = {                    # spelling/format variants -> canonical surface
    "ci cd": "CI/CD", "ci-cd": "CI/CD", "cicd": "CI/CD", "ci/cd": "CI/CD",
    "node js": "Node.js", "nodejs": "Node.js", "power bi": "Power BI",
    "powerbi": "Power BI", "javascript": "JavaScript", "js": "JavaScript",
    "postgres": "PostgreSQL", "k8s": "Kubernetes", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "scikit-learn": "scikit-learn",
    "rest api": "REST API", "restful": "REST API", "rest apis": "REST API",
    "ms excel": "Excel", "microsoft excel": "Excel", "spark": "Apache Spark",
    "kafka": "Apache Kafka", "airflow": "Apache Airflow",
}
# Synonym merge: collapse variants of ONE concept to a single canonical entry.
_DW, _IaC, _MSO = "Data Warehouse", "Infrastructure as Code", "Microsoft Office"
SYNONYMS = {
    "data warehouses": _DW, "data warehousing": _DW, "data modeling": "Data Modeling",
    "data modelling": "Data Modeling", "data models": "Data Modeling",
    "data analytics": "Data Analysis", "genai": "Generative AI", "gen ai": "Generative AI",
    "artificial intelligence": "AI", "natural language processing": "NLP",
    "large language models": "LLM", "llms": "LLM", "iac": _IaC, "infrastructure-as-code": _IaC,
    "pci-dss": "PCI DSS", "ms office": _MSO, "microsoft office suite": _MSO,
    "microsoft power bi": "Power BI", "google cloud": "GCP", "google cloud platform": "GCP",
    "microsoft azure": "Azure", "oracle cloud infrastructure": "OCI", "oracle cloud": "OCI",
    "elk stack": "ELK", "nosql databases": "NoSQL", "relational databases": "RDBMS",
    "ips/ids": "IDS/IPS", "intrusion detection systems": "IDS/IPS", "vpns": "VPN",
    "vpcs": "VPC", "golang": "Go", "ms sql server": "SQL Server", "restful apis": "REST API",
    "microservices architecture": "microservices", "security+": "CompTIA Security+",
    "cloud services": "cloud platforms", "cloud technologies": "cloud platforms",
    "gitlab ci/cd": "GitLab CI",
}


def normalize_skill(raw: str) -> str | None:
    """Lowercase-key normalize; return canonical surface form or None to drop."""
    s = str(raw).strip().strip(".,;:!()[]{}\"'`").strip()
    s = re.sub(r"\s+", " ", s)
    key = s.lower()
    if len(key) < 2 or key.isdigit():
        return None
    canon = ALIASES.get(key) or SYNONYMS.get(key)
    if canon:
        return canon
    if (key in SPOKEN_LANGUAGES or key in SOFT_SKILLS or key in PLATFORMS_BLOCK
            or key in GENERIC_BLOCK):
        return None
    return s    # keep original casing; aggregation dedups case-insensitively


def load_company_names() -> set:
    """Lowercased company names to filter out (safety net)."""
    try:
        col = pd.read_csv(JOBS_RAW_CSV, usecols=["company"])["company"]
    except Exception:
        return set()
    return {c.strip().lower() for c in col.dropna() if isinstance(c, str)}


def focus_text(text: str, max_chars: int = MAX_FOCUS_CHARS) -> str:
    """Front-load the requirements/skills section so the gold leads the prompt."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    m = GOLD_HEADER_RE.search(text)
    if m and m.start() > 400:                  # only reorder if real fluff precedes
        text = text[m.start():] + "  " + text[:m.start()]
    return text[:max_chars]


_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def extract_skills_one(description: str, timeout: int = 180) -> list[str]:
    """Call Ollama once; return raw (un-normalized) skill strings."""
    prompt = SKILL_PROMPT.format(description=focus_text(description))
    resp = get_session().post(
        OLLAMA_API_GENERATE,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 400},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _parse_skills(resp.json()["response"])


def _parse_skills(raw: str) -> list[str]:
    """Parse {"skills":[...]} or a bare [...]; tolerate fences/prose."""
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, raw, re.DOTALL)
        if not m:
            continue
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), None)
        if isinstance(data, list):
            return [str(s) for s in data
                    if isinstance(s, (str, int, float)) and str(s).strip()]
    return []


def load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("skills") is not None:      # errors are retried
                        done.add(rec["job_id"])
                except Exception:
                    pass
    return done


def sample_stratified(df, n, class_col="final_category"):
    counts = df[class_col].value_counts()
    per_class = (counts / len(df) * n).astype(int).clip(lower=1)
    while per_class.sum() > n:
        per_class[per_class.idxmax()] -= 1
    parts = []
    for cls, cnt in per_class.items():
        sub = df[df[class_col] == cls]
        parts.append(sub if len(sub) <= cnt else sub.sample(cnt, random_state=42))
    return pd.concat(parts).sample(frac=1, random_state=42)


def aggregate_and_write(min_freq: int) -> None:
    companies = load_company_names()
    doc_freq: Counter = Counter()          # lowercase key -> #docs it appears in
    surfaces: dict[str, Counter] = {}      # lowercase key -> {surface form: count}
    n_docs = n_empty = 0
    with OUTPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("skills") is None:
                continue
            n_docs += 1
            seen = set()
            for raw in rec["skills"]:
                norm = normalize_skill(raw)
                if norm is None or norm.lower() in companies:
                    continue
                key = norm.lower()
                seen.add(key)
                surfaces.setdefault(key, Counter())[norm] += 1
            if not seen:
                n_empty += 1
            doc_freq.update(seen)           # document frequency (once per doc)
    # Dedup case variants: one entry per key, shown in its most common casing.
    vocab = sorted((surfaces[k].most_common(1)[0][0]
                    for k, c in doc_freq.items() if c >= min_freq), key=str.lower)
    print(f"\nDocs with skills: {n_docs:,} (empty: {n_empty}) | unique: {len(doc_freq):,} | kept (>= {min_freq}): {len(vocab):,}")
    body = ",\n    ".join(repr(s) for s in vocab)
    OUTPUT_VOCAB.write_text(
        '"""Auto-generated technical skill vocabulary (LLM-extracted).\n\n'
        "Regenerate with: python scripts/build_skill_vocab_llm.py\n"
        f'Source: {OUTPUT_JSONL.name}; doc-frequency threshold >= {min_freq}.\n"""\n\n'
        f"SKILLS_VOCABULARY: list[str] = [\n    {body},\n]\n\n"
        f"SKILL_ALIASES = {json.dumps(ALIASES, indent=4, sort_keys=True)}\n\n"
        f"SKILL_SYNONYMS = {json.dumps(SYNONYMS, indent=4, sort_keys=True)}\n\n\n"
        "def get_skill_vocabulary() -> list[str]:\n"
        '    """Return the curated skill vocabulary (sorted)."""\n'
        "    return list(SKILLS_VOCABULARY)\n",
        encoding="utf-8")
    print(f"Vocabulary saved -> {OUTPUT_VOCAB}\nFirst 30: {', '.join(vocab[:30])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = all tech descriptions")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-freq", type=int, default=MIN_SKILL_FREQ)
    ap.add_argument("--reset", action="store_true", help="delete progress, start over")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip extraction, just rebuild skill_vocab.py from JSONL")
    args = ap.parse_args()
    if args.aggregate_only:
        if not OUTPUT_JSONL.exists():
            sys.exit(f"No JSONL to aggregate at {OUTPUT_JSONL}")
        aggregate_and_write(args.min_freq)
        return
    df = pd.read_csv(DATA_PATH)
    df = df[df["final_category"].isin(TECH_CLASSES)].copy()
    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample} (stratified by role) ...")
        df = sample_stratified(df, args.sample)
    if args.reset and OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
    done = load_done(OUTPUT_JSONL)
    todo = df[~df["job_id"].isin(done)].copy()
    print(f"Model: {MODEL} | workers: {args.workers} | tech: {len(df):,} | "
          f"done: {len(done):,} | to do: {len(todo):,}")
    try:
        requests.get(OLLAMA_API_TAGS, timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running?\n  {e}")
    if len(todo):
        OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        write_lock = threading.Lock()
        counter = {"n": 0, "err": 0, "empty": 0}
        started = time.time()
        out_f = OUTPUT_JSONL.open("a", encoding="utf-8")

        def work(row):
            try:
                skills = extract_skills_one(row.clean_description)
                rec = {"job_id": row.job_id, "category": row.final_category,
                       "skills": skills}
            except Exception as e:
                rec = {"job_id": row.job_id, "skills": None,
                       "error": f"{type(e).__name__}: {e}"[:160]}
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter["n"] += 1
                if rec.get("skills") is None:
                    counter["err"] += 1
                elif not rec["skills"]:
                    counter["empty"] += 1
                if counter["n"] % 10 == 0:
                    out_f.flush()
                n = counter["n"]
                if n % 20 == 0 or n == len(todo):
                    el = time.time() - started
                    rate = n / el if el else 0
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"[{n:>5}/{len(todo)}] {rate:4.2f} rows/s  ETA {eta:5.1f}m  "
                          f"errors {counter['err']}  empty {counter['empty']}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, r) for r in todo.itertuples(index=False)]
            for _ in as_completed(futures):
                pass
        out_f.close()
        el = (time.time() - started) / 60
        print(f"\nExtraction done: {counter['n']:,} rows in {el:.1f} min "
              f"({counter['err']} errors, {counter['empty']} empty).")
    else:
        print("All descriptions already extracted; aggregating.")
    aggregate_and_write(args.min_freq)


if __name__ == "__main__":
    main()
