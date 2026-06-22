"""
Adjudicate skill-implication mappings from two LLM runs (7B + 32B).

Uses the same pattern as the job-labelling pipeline:
1. Load both JSONL files.
2. For each term, compare the two skill-lists via Jaccard similarity.
   - High agreement (Jaccard ≥ 0.7) → keep the 7B list (it is the stronger model).
   - Low agreement or one list empty → call the 32B judge to produce a
     consolidated list.
3. Post-process: filter against SKILLS_VOCABULARY, deduplicate, canonicalise
   casing.
4. Save scripts/skill_mappings.py containing TOOL_SKILL_MAP.

Usage:
    python scripts/adjudicate_skill_mappings.py --workers 4
"""

import json
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
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")
JUDGE_MODEL = "qwen2.5:32b-instruct-q4_K_M"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.skill_vocab import SKILLS_VOCABULARY

FILE_7B = PROJECT_ROOT / "data" / "processed" / "skill_mappings_7b.jsonl"
FILE_32B = PROJECT_ROOT / "data" / "processed" / "skill_mappings_32b.jsonl"
OUTPUT_PY = PROJECT_ROOT / "scripts" / "skill_mappings.py"
OUTPUT_JUDGE_JSONL = PROJECT_ROOT / "data" / "processed" / "skill_judge_results.jsonl"

JACCARD_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Judge prompt – proven in testing
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are a technical skills expert. Given a tool or certification and two suggested lists of implied skills from different models, produce the single best consolidated list.

Step 1 — Reason about which skills are genuinely implied by the term. Consider:
- Are any skills in the lists irrelevant or misattributed? Remove them.
- Are any important skills missing from both lists? If so, add them.
- Are there near-duplicates? Prefer the most specific and widely recognized canonical name.

Step 2 — Output your final consolidated list.

Return ONLY a JSON object with:
{{
  "consolidated_skills": ["skill1", "skill2", ...],
  "reasoning": "<one short sentence explaining key decisions>",
  "certainty": "high|medium|low"
}}

Term: {term}
List A: {list_a}
List B: {list_b}

JSON response:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    """Return {term: [skills]} dict from a JSONL file."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            term = rec["term"]
            skills = rec.get("implied_skills") or []
            data[term] = [s.strip() for s in skills if s.strip()]
    return data


def jaccard(list_a, list_b):
    """Jaccard similarity between two sets of case‑folded skills."""
    a = {s.lower() for s in list_a}
    b = {s.lower() for s in list_b}
    if not a and not b:
        return 1.0          # both empty → agree
    if not a or not b:
        return 0.0          # one empty → disagree
    return len(a & b) / len(a | b)


from scripts.skill_vocab import SKILL_ALIASES, SKILL_SYNONYMS

def resolve_to_canonical(term: str) -> str | None:
    """Map a term through aliases and synonyms to its canonical vocabulary form."""
    key = term.strip().lower()
    # Direct match in vocabulary
    if key in {s.lower() for s in SKILLS_VOCABULARY}:
        return term  # Keep original casing from the judge (will be fixed later)
    # Check aliases
    if key in SKILL_ALIASES:
        return SKILL_ALIASES[key]
    # Check synonyms
    if key in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[key]
    # Try case‑insensitive match in synonyms/aliases
    for d in (SKILL_ALIASES, SKILL_SYNONYMS):
        for k, v in d.items():
            if k.lower() == key:
                return v
    return None

def canonicalise(skills):
    # Direct imports to avoid any module cache issues
    from scripts.skill_vocab import SKILLS_VOCABULARY, SKILL_ALIASES, SKILL_SYNONYMS

    vocab_lower = {s.lower(): s for s in SKILLS_VOCABULARY}
    alias_lower  = {k.lower(): v for k, v in SKILL_ALIASES.items()}
    synonym_lower = {k.lower(): v for k, v in SKILL_SYNONYMS.items()}

    seen = set()
    out = []
    for s in skills:
        key = s.strip().lower()
        # 1. Is it already a canonical form?
        if key in vocab_lower:
            canon = vocab_lower[key]
        # 2. Check aliases
        elif key in alias_lower:
            canon = alias_lower[key]
        # 3. Check synonyms
        elif key in synonym_lower:
            canon = synonym_lower[key]
        else:
            # Not recognised at all → discard
            continue

        if canon.lower() not in seen:
            seen.add(canon.lower())
            out.append(canon)

    return sorted(out, key=str.lower)


# ---------------------------------------------------------------------------
# Thread‑safe judge
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def call_judge(term, list_a, list_b, timeout=60):
    session = get_session()
    prompt = JUDGE_PROMPT.format(term=term, list_a=json.dumps(list_a), list_b=json.dumps(list_b))
    resp = session.post(
        OLLAMA_API_GENERATE,
        json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": "15m",
            "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["response"]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group()
    result = json.loads(raw)
    skills = result.get("consolidated_skills", [])
    if not isinstance(skills, list):
        skills = []
    return [s.strip() for s in skills if s.strip()]


def load_done_judge():
    done = set()
    if OUTPUT_JUDGE_JSONL.exists():
        with open(OUTPUT_JUDGE_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("skills") is not None:
                        done.add(rec["term"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reset", action="store_true", help="Delete previous judge results")
    args = ap.parse_args()

    print("Loading model outputs ...")
    data7b = load_jsonl(FILE_7B)
    data32b = load_jsonl(FILE_32B)
    terms = sorted(set(data7b.keys()) & set(data32b.keys()), key=str.lower)
    print(f"Terms to adjudicate: {len(terms):,}")

    # ---- Decide which terms need the judge ----
    need_judge = []
    final = {}
    for term in terms:
        a = data7b.get(term, [])
        b = data32b.get(term, [])
        if jaccard(a, b) >= JACCARD_THRESHOLD:
            # High agreement — use the 7B list
            final[term] = a
        elif a and not b:
            # Only 7B has data — use it
            final[term] = a
        else:
            # Disagreement or both empty — needs judge
            need_judge.append((term, a, b))

    print(f"Agreed (direct use): {len(final):,}")
    print(f"Need judge:          {len(need_judge):,}")

    # ---- Process disagreements with the judge ----
    if args.reset and OUTPUT_JUDGE_JSONL.exists():
        OUTPUT_JUDGE_JSONL.unlink()
    done_judge = load_done_judge()
    todo = [(term, a, b) for term, a, b in need_judge if term not in done_judge]
    print(f"Already judged: {len(done_judge):,}  |  To judge: {len(todo):,}")

    if todo:
        write_lock = threading.Lock()
        counter = {"n": 0, "err": 0}
        started = time.time()
        out_f = open(OUTPUT_JUDGE_JSONL, "a", encoding="utf-8")

        def work(item):
            term, list_a, list_b = item
            try:
                skills = call_judge(term, list_a, list_b)
                rec = {"term": term, "skills": skills}
            except Exception as e:
                rec = {"term": term, "skills": None,
                       "error": f"{type(e).__name__}: {e}"[:160]}
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter["n"] += 1
                if rec.get("skills") is None:
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
            futures = [ex.submit(work, item) for item in todo]
            for _ in as_completed(futures):
                pass

        out_f.close()
        elapsed = (time.time() - started) / 60
        print(f"\nJudge finished: {counter['n']} terms in {elapsed:.1f} min "
              f"({counter['err']} errors).")

    # ---- Merge judge results ----
    if OUTPUT_JUDGE_JSONL.exists():
        with open(OUTPUT_JUDGE_JSONL, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                term = rec["term"]
                skills = rec.get("skills")
                if skills is not None:
                    final[term] = skills
                else:
                    # Fallback: use whichever model list is non‑empty
                    final[term] = data7b.get(term) or data32b.get(term) or []

    # ---- Post-process: canonicalise every list ----
    print("Canonicalising against skill vocabulary ...")
    canonical_map = {}
    for term, skills in final.items():
        clean = canonicalise(skills)
        if clean:
            canonical_map[term] = clean

    print(f"Final mappings: {len(canonical_map):,} terms")

    # ---- Save ----
    body = json.dumps(canonical_map, indent=4)
    OUTPUT_PY.write_text(
        f'"""Auto‑generated skill mappings (adjudicated).\n\n'
        f'Regenerate with: python scripts/adjudicate_skill_mappings.py\n'
        f'"""\n\n'
        f'TOOL_SKILL_MAP = {body}\n\n'
        f'CERT_SKILL_MAP = TOOL_SKILL_MAP\n\n'
        f'def get_tool_skill_map() -> dict[str, list[str]]:\n'
        f'    return TOOL_SKILL_MAP\n\n'
        f'def get_cert_skill_map() -> dict[str, list[str]]:\n'
        f'    return TOOL_SKILL_MAP\n',
        encoding="utf-8",
    )
    print(f"Saved → {OUTPUT_PY}")


if __name__ == "__main__":
    import argparse
    main()