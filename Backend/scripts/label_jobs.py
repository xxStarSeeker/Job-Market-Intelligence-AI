"""
Label every job with a local LLM (Ollama qwen2.5:32b).
Closed‑set classification into predefined tech roles + key extraction + reasoning.
Uses a thread pool for speed (set OLLAMA_NUM_PARALLEL=4 before starting Ollama).

Usage:
  python -m Backend.scripts.label_jobs --workers 4 --out labels/labels_closed.jsonl
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from Backend import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = config.OLLAMA_URL  # e.g. "http://localhost:11434"
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")
OLLAMA_API_TAGS = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/tags")
DEFAULT_MODEL = config.OLLAMA_MODEL          # "qwen2.5:32b-instruct-q4_K_M"

# ---------------------------------------------------------------------------
# Closed set of job categories – EDIT THIS LIST AS NEEDED
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Data Analyst",
    "Data Engineer",
    "ML Engineer",
    "AI Engineer",
    "Backend Developer",
    "Frontend Developer",
    "DevOps Engineer",
    "Software Engineer",          # optional, remove if you want strict separation
    "Other Tech",                 # catch‑all for tech roles not listed above
    "Non-Tech",                   # everything else (sales, finance, etc.)
]

CATEGORIES_BULLETED = "\n".join(f"- {c}" for c in CATEGORIES)

# ---------------------------------------------------------------------------
# Prompt – Chain‑of‑Thought with reasoning first
# ---------------------------------------------------------------------------
PROMPT = """You are an expert job classifier specialising in technology roles.

Your task is to read a job DESCRIPTION and assign it to EXACTLY ONE of the following predefined categories.
The original job title is HIDDEN on purpose – rely ONLY on the actual duties, tools, and skills mentioned in the description.

AVAILABLE CATEGORIES:
{categories}

RULES:
- Read the job description carefully.
- Ignore company names and locations; focus on what the person will DO day-to-day.
- Choose the single best matching category from the list above.
- If the job involves data pipelines, ETL, data warehouses → "Data Engineer".
- If it involves building dashboards, SQL queries, reporting, business insights → "Data Analyst".
- If it involves training/deploying machine learning models (scikit-learn, TensorFlow, etc.) → "ML Engineer".
- If it involves working with LLMs, NLP, computer vision, AI APIs, agents, RAG → "AI Engineer".
- If it involves server-side coding, APIs, databases (Python, Java, Node, etc.) → "Backend Developer".
- If it involves UI, JavaScript, React, Vue, CSS → "Frontend Developer".
- If it involves CI/CD, Docker, Kubernetes, infrastructure, cloud ops → "DevOps Engineer".
- If it involves general software development or ambiguous "Software Engineer" roles → "Software Engineer".
- If it is a technology role that does NOT fit any of the above (e.g., QA, cybersecurity, network admin, IT support) → "Other Tech".
- If the job is clearly NOT a technology/IT role (e.g., sales, accounting, healthcare, engineering, hospitality) → "Non-Tech".

Your response MUST be a single JSON object with EXACTLY this structure (note: reasoning MUST come first so you can think before deciding):

{{
  "reasoning": "<Think step-by-step. 1-2 sentences explaining why based on the rules.>",
  "category": "<exact category from the list>",
  "certainty": "high|medium|low",
  "key_skills": ["Python", "SQL", "Spark"]   // example – replace with actual skills
}}

"certainty" guide:
- high   – The description leaves no doubt; the role is clearly this category.
- medium – Mostly matches, but some ambiguity (e.g., a "Data Specialist" that could be Analyst or Engineer).
- low    – You are guessing; the description is vague or overlaps multiple categories.

Job description:
{description}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truncate(text, n=2500):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n] + "..."


def label_one(model: str, description: str, timeout=120) -> dict:
    """
    Send one job description to the LLM.
    Returns dict with: category, certainty, reasoning, key_skills.
    """
    session = get_session()
    prompt = PROMPT.format(
        categories=CATEGORIES_BULLETED,
        description=truncate(description if isinstance(description, str) else ""),
    )
    resp = session.post(
        OLLAMA_API_GENERATE,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 300,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]

    # Robust JSON extraction – handles occasional markdown fences
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group()
    parsed = json.loads(raw)

    # Category – ensure it's one of our list
    category = str(parsed.get("category", "")).strip()
    if category not in CATEGORIES:
        for c in CATEGORIES:
            if c.lower() == category.lower():
                category = c
                break
        else:
            category = "Other Tech"

    # Certainty
    certainty_raw = str(parsed.get("certainty", "low")).strip().lower()
    if certainty_raw not in ("high", "medium", "low"):
        certainty_raw = "low"

    # Reasoning
    reasoning = str(parsed.get("reasoning", "")).strip()

    # Key skills – robust parsing for both array and string
    skills_raw = parsed.get("key_skills", [])
    if isinstance(skills_raw, list):
        key_skills = ", ".join(str(s).strip() for s in skills_raw)
    else:
        key_skills = str(skills_raw).strip()

    return {
        "category": category,
        "certainty": certainty_raw,
        "reasoning": reasoning[:300],
        "key_skills": key_skills[:300],
    }


# Thread‑local session (thread‑safe)
_thread_local = threading.local()

def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# Resume handling
# ---------------------------------------------------------------------------
def load_done(out_path: Path) -> set:
    """Return set of job_ids that were successfully labeled (not errors)."""
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("llm_certainty") != "ERROR":
                        done.add(rec["job_id"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def write_csv_from_jsonl(jsonl_path: Path, source_csv: Path, csv_path: Path):
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return
    labels = pd.DataFrame(rows)
    try:
        src = pd.read_csv(source_csv, usecols=["job_id", "company", "location"], low_memory=False)
        labels = labels.merge(src, on="job_id", how="left")
    except Exception:
        pass
    cols = [
        "job_id", "job_title", "company", "location",
        "llm_category", "llm_certainty", "llm_reasoning", "llm_key_skills"
    ]
    out = labels[[c for c in cols if c in labels.columns]].rename(
        columns={
            "llm_category": "category",
            "llm_certainty": "certainty",
            "llm_reasoning": "reasoning",
            "llm_key_skills": "key_skills",
        }
    )
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV written: {csv_path}  ({len(out):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.JOBS_CLEAN_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests (match OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    csv_path, out_path = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path, low_memory=False)
    if args.limit:
        df = df.head(args.limit).copy()
    done = load_done(out_path)
    todo = df[~df["job_id"].isin(done)].copy()
    print(f"Model: {args.model} | workers: {args.workers}")
    print(f"Loaded {len(df):,} rows | already done {len(done):,} | to label {len(todo):,}")

    # Health check
    try:
        requests.get(OLLAMA_API_TAGS, timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running?\n  {e}")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    # Thread‑safe writer
    write_lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    started = time.time()
    out_f = out_path.open("a", encoding="utf-8")

    def work(row):
        try:
            label = label_one(args.model, row.job_description)   # title is NOT passed
            rec = {
                "job_id": row.job_id,
                "job_title": row.job_title,                    # kept for reference only
                "llm_category": label["category"],
                "llm_certainty": label["certainty"],
                "llm_reasoning": label["reasoning"],
                "llm_key_skills": label["key_skills"],
            }
        except Exception as e:
            rec = {
                "job_id": row.job_id,
                "job_title": row.job_title,
                "llm_category": "ERROR",
                "llm_certainty": "ERROR",
                "llm_reasoning": "",
                "llm_key_skills": "",
                "error": f"{type(e).__name__}: {e}"[:160],
            }
        with write_lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if rec.get("llm_certainty") == "ERROR":
                counter["err"] += 1
            if counter["n"] % 10 == 0:
                out_f.flush()
            n = counter["n"]
            if n % 20 == 0 or n == len(todo):
                el = time.time() - started
                rate = n / el if el > 0 else 0
                eta = (len(todo) - n) / rate / 60 if rate else 0
                print(
                    f"[{n:>6}/{len(todo)}] {rate:4.1f} rows/s  "
                    f"ETA {eta:5.1f}m  errors {counter['err']}  "
                    f"last: {rec['llm_category']}",
                    flush=True,
                )
        return rec

    rows = list(todo.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, r) for r in rows]
        for _ in as_completed(futures):
            pass

    out_f.flush()
    out_f.close()
    el = (time.time() - started) / 60
    print(f"\nDone. {counter['n']:,} rows in {el:.1f} min ({counter['err']} errors). -> {out_path}")

    # Write CSV
    csv_out = out_path.with_suffix(".csv")
    write_csv_from_jsonl(out_path, csv_path, csv_out)


if __name__ == "__main__":
    main()