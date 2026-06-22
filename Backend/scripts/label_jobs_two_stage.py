"""
Resolve label disagreements using an LLM as an impartial judge.
- Reads disagreements.csv (from the earlier comparison script)
- Merges with original job descriptions
- For each disagreed job, asks the LLM to pick the best label (from the two suggestions, 
  or propose a new one)
- Outputs a CSV with the final label for every job (agreed + resolved)

Usage: python resolve_disagreements_open.py
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

# =============================================================================
# CONFIGURATION – adjust paths if needed
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")
MODEL = "qwen2.5:32b-instruct-q4_K_M"

DISAGREEMENTS_FILE = "disagreements.csv"          # from the earlier comparison script
LABEL_FILE_32B = "labels_all.csv"                 # needed for agreed jobs
LABEL_FILE_8B  = "labels_all1.csv"                # needed for agreed jobs
JOBS_FILE      = "job_data_saudi_clean.csv"       # original job descriptions

OUTPUT_FINAL   = "final_labels_resolved.csv"      # final label per job
OUTPUT_LLM_RES = "disagreements_llm_resolved.jsonl"  # raw LLM answers (for debugging)
# =============================================================================

# ---------------------------------------------------------------------------
# Prompt – open vocabulary (no fixed list)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an expert job classifier.

Read the job description and two existing label suggestions made by two different models.

Model 1 suggested: {label_32b}
Model 2 suggested: {label_8b}

Your task:
- Choose the single most accurate job category label for this description.
- If one of the two existing suggestions is already correct, use it EXACTLY as written.
- If neither is ideal, output a BETTER label that accurately captures the role.
  * Keep it concise (1-4 words)
  * No seniority words (Senior/Junior/Lead)
  * No company names or locations
  * Prefer standard, real-world job titles

Return ONLY a JSON object with:
{{
  "final_category": "<chosen label>",
  "reasoning": "<one short sentence explaining your choice>",
  "certainty": "high|medium|low"
}}

Job description:
{description}"""

# ---------------------------------------------------------------------------
# Thread‑safe helpers
# ---------------------------------------------------------------------------
def truncate(text, n=2500):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n] + "..."

_thread_local = threading.local()
def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session

def judge_one(description, label_32b, label_8b, timeout=120):
    session = get_session()
    prompt = JUDGE_PROMPT.format(
        label_32b=label_32b,
        label_8b=label_8b,
        description=truncate(description),
    )
    resp = session.post(
        OLLAMA_API_GENERATE,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 200},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group()
    parsed = json.loads(raw)
    final_cat = str(parsed.get("final_category", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    certainty = str(parsed.get("certainty", "medium")).strip().lower()
    if certainty not in ("high", "medium", "low"):
        certainty = "medium"
    # Ensure we return a non‑empty label
    if not final_cat:
        final_cat = label_32b  # fallback to 32B's suggestion
    return {"final_category": final_cat, "reasoning": reasoning[:300], "certainty": certainty}

# ---------------------------------------------------------------------------
# Resume handling for the LLM output (JSONL)
# ---------------------------------------------------------------------------
def load_done_llm(out_path):
    done = set()
    if Path(out_path).exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("final_category") != "ERROR":
                        done.add(rec["job_id"])
                except:
                    pass
    return done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load the disagreements table
    dis_df = pd.read_csv(DISAGREEMENTS_FILE)
    # Load original job descriptions
    jobs_df = pd.read_csv(JOBS_FILE, usecols=["job_id", "job_description"])
    # Merge
    dis_df = dis_df.merge(jobs_df, on="job_id", how="inner")
    if dis_df.empty:
        print("No disagreements found or missing descriptions.")
        return

    # ---- Process disagreements with LLM ----
    done_llm = load_done_llm(OUTPUT_LLM_RES)
    todo = dis_df[~dis_df["job_id"].isin(done_llm)]
    print(f"Total disagreements: {len(dis_df)} | Already resolved: {len(done_llm)} | To process: {len(todo)}")

    if len(todo) > 0:
        write_lock = threading.Lock()
        counter = {"n": 0, "err": 0}
        started = time.time()
        out_f = open(OUTPUT_LLM_RES, "a", encoding="utf-8")

        def work(row):
            try:
                result = judge_one(row.job_description, row.category_32b, row.category_8b)
                rec = {
                    "job_id": row.job_id,
                    "job_title": row.job_title,
                    "category_32b": row.category_32b,
                    "category_8b": row.category_8b,
                    "final_category": result["final_category"],
                    "reasoning": result["reasoning"],
                    "certainty": result["certainty"],
                }
            except Exception as e:
                rec = {
                    "job_id": row.job_id,
                    "job_title": row.job_title,
                    "category_32b": row.category_32b,
                    "category_8b": row.category_8b,
                    "final_category": "ERROR",
                    "reasoning": "",
                    "certainty": "",
                    "error": str(e)[:200],
                }
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter["n"] += 1
                if rec["final_category"] == "ERROR":
                    counter["err"] += 1
                if counter["n"] % 10 == 0:
                    out_f.flush()
                if counter["n"] % 20 == 0 or counter["n"] == len(todo):
                    el = time.time() - started
                    rate = counter["n"] / el if el > 0 else 0
                    eta = (len(todo) - counter["n"]) / rate / 60 if rate > 0 else 0
                    print(f"[{counter['n']:>5}/{len(todo)}] {rate:4.1f} rows/s  "
                          f"ETA {eta:5.1f}m  errors {counter['err']}  "
                          f"last: {rec.get('final_category','')}",
                          flush=True)

        rows = list(todo.itertuples(index=False))
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(work, r) for r in rows]
            for _ in as_completed(futures):
                pass
        out_f.close()
        elapsed = (time.time() - started) / 60
        print(f"\nLLM resolution complete. {counter['n']} processed in {elapsed:.1f} min ({counter['err']} errors).")

    # ---- Merge resolved disagreements with agreed jobs ----
    # Load LLM results and map job_id -> final_category
    llm_df = pd.read_json(OUTPUT_LLM_RES, lines=True)
    llm_map = dict(zip(llm_df["job_id"], llm_df["final_category"]))

    # Load the full label files
    df32 = pd.read_csv(LABEL_FILE_32B)
    df8  = pd.read_csv(LABEL_FILE_8B)
    for df in [df32, df8]:
        if "category" not in df.columns and "llm_category" in df.columns:
            df.rename(columns={"llm_category": "category"}, inplace=True)

    # Merge to find agreed jobs (inner join on job_id)
    merged = pd.merge(df32[["job_id", "category"]], df8[["job_id", "category"]],
                      on="job_id", how="inner", suffixes=("_32b", "_8b"))
    # For jobs where both labels agree, use that label
    agreed = merged[merged["category_32b"] == merged["category_8b"]].copy()
    agreed["final_category"] = agreed["category_32b"]

    # For disagreed jobs, pull the LLM resolution
    disagreed_ids = set(llm_map.keys())
    # If some jobs weren't resolved (e.g., errors), fallback to 32B label
    for job_id in disagreed_ids:
        if job_id not in agreed["job_id"].values:  # in case it's missing from agreed
            # We'll create a placeholder; but normally all should exist in merged
            pass

    # Combine: agreed jobs + resolved disagreements
    resolved_jobs = pd.DataFrame({
        "job_id": list(llm_map.keys()),
        "final_category": [llm_map[jid] for jid in llm_map.keys()]
    })

    # Ensure we have job_title etc. – grab from df32 (or original)
    # We'll do a left join with the original jobs to get titles
    jobs_full = pd.read_csv(JOBS_FILE, usecols=["job_id", "job_title"])
    final_df = pd.concat([agreed[["job_id", "final_category"]], resolved_jobs], ignore_index=True)
    final_df = final_df.merge(jobs_full, on="job_id", how="left")

    # Save the final unified labels
    final_df.to_csv(OUTPUT_FINAL, index=False, columns=["job_id", "job_title", "final_category"])
    print(f"Final labels saved to {OUTPUT_FINAL}")
    print(f"Total jobs: {len(final_df)}")
    print(f"Agreed: {len(agreed)} | Resolved by LLM: {len(resolved_jobs)}")

if __name__ == "__main__":
    main()