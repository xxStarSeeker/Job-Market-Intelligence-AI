"""
Extract implied skills from tool/certification terms using an LLM.
Supports --model and --out for multi-model runs.
Resumable, thread-safe.

Usage:
    python scripts/build_skill_mappings_llm.py --model qwen2.5:32b-instruct-q4_K_M --out skill_mappings_32b.jsonl
    python scripts/build_skill_mappings_llm.py --model qwen2.5:8b-instruct       --out skill_mappings_8b.jsonl
    python scripts/build_skill_mappings_llm.py --model qwen2.5:7b-instruct       --out skill_mappings_7b.jsonl
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.skill_vocab import SKILLS_VOCABULARY

# ---------------------------------------------------------------------------
# Prompt – sentence completion to avoid model refusal
# ---------------------------------------------------------------------------
MAPPING_PROMPT = """You are an expert technical skills analyst.
Given a specific tool, technology, or certification, list the higher‑level skills, competencies, and knowledge domains that proficiency with it implies.

Rules:
- Use 1‑4 word skill names (e.g., "Data Governance", "Cloud Architecture", "Big Data Processing").
- Be comprehensive: include everything a certification exam covers, or what professional experience with the tool demonstrates.
- Return ONLY a valid JSON array of strings. No other text.

Example 1:
Tool: Apache Spark
Skills: ["Big Data Processing", "Distributed Computing", "Data Engineering", "ETL", "Scala Programming"]

Example 2:
Tool: AWS
Skills: ["Cloud Computing", "Cloud Architecture", "Cloud Security", "Infrastructure as Code", "Serverless Computing"]

Example 3:
Tool: CDMP
Skills: ["Data Governance", "Data Architecture", "Data Quality", "Metadata Management", "Data Stewardship"]

Example 4:
Tool: API testing
Skills: ["Software Testing", "Test Automation", "API Design", "Quality Assurance", "Integration Testing"]

Example 5:
Tool: AI
Skills: ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Model Deployment"]

Tool: {term}
Skills:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_thread_local = threading.local()

def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session

def map_one(term: str, model: str, timeout: int = 30) -> list[str]:
    """Call Ollama and return the implied skills."""
    session = get_session()
    prompt = MAPPING_PROMPT.format(term=term)
    resp = session.post(
        OLLAMA_API_GENERATE,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "15m",
            "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300},  # more room
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]

    # Robust JSON extraction – handle both arrays and objects
    # First, try to find a clean array
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group()
    else:
        # If no array found, try to find an object (the model sometimes returns that)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group()

    # Repair truncated JSON: close unclosed quotes and brackets
    if raw.count('"') % 2 != 0:
        raw += '"'
    if raw.count('[') > raw.count(']'):
        raw += ']'

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            skills = [str(s).strip() for s in data if isinstance(s, str) and s.strip()]
        elif isinstance(data, dict):
            skills = []
            for k, v in data.items():
                if isinstance(k, str) and k.strip():
                    skills.append(k.strip())
                if isinstance(v, str) and v.strip():
                    skills.append(v.strip())
            # Deduplicate while preserving order
            seen = set()
            skills = [s for s in skills if not (s.lower() in seen or seen.add(s.lower()))]
        else:
            skills = []
    except json.JSONDecodeError:
        skills = []

    return skills

def load_done(out_path):
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["term"])
                except Exception:
                    pass
    return done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--out", required=True, help="Output JSONL file path")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    model = args.model
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / "data" / "processed" / out_path

    terms = sorted(SKILLS_VOCABULARY)
    print(f"Model: {model}  |  Terms: {len(terms)}  |  Output: {out_path}")

    if args.reset and out_path.exists():
        out_path.unlink()

    done = load_done(out_path)
    todo = [t for t in terms if t not in done]
    print(f"Done: {len(done):,}  |  To do: {len(todo):,}")

    if not todo:
        print("All terms already processed.")
        return

    write_lock = threading.Lock()
    counter = {"n": 0, "err": 0}
    started = time.time()
    out_f = out_path.open("a", encoding="utf-8")

    def work(term):
        try:
            implied = map_one(term, model)
            rec = {"term": term, "implied_skills": implied}
        except Exception as e:
            rec = {"term": term, "implied_skills": None,
                   "error": f"{type(e).__name__}: {e}"[:160]}
        with write_lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if rec.get("implied_skills") is None:
                counter["err"] += 1
            if counter["n"] % 10 == 0:
                out_f.flush()
            n = counter["n"]
            if n % 20 == 0 or n == len(todo):
                el = time.time() - started
                rate = n / el if el else 0
                eta = (len(todo) - n) / rate / 60 if rate else 0
                print(f"[{n:>4}/{len(todo)}] {rate:4.1f} rows/s  "
                      f"ETA {eta:5.1f}m  errors {counter['err']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, t) for t in todo]
        for _ in as_completed(futures):
            pass

    out_f.close()
    elapsed = (time.time() - started) / 60
    print(f"\nFinished: {counter['n']} terms in {elapsed:.1f} min ({counter['err']} errors).")

if __name__ == "__main__":
    main()