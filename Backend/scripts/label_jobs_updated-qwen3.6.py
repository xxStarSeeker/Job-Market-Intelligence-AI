"""
Two-stage job labelling that avoids title bias.
Optimized for high-precision, high-throughput asynchronous execution.

Usage:
  python -m Backend.scripts.label_jobs_two_stage --out Backend/labels/labels_all.jsonl --model qwen3.5:27b-q6_K
"""

import argparse
import json
import re
import sys
import time
import asyncio
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from Backend import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = config.OLLAMA_URL
OLLAMA_API_GENERATE = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/generate")
OLLAMA_API_TAGS = urljoin(OLLAMA_BASE_URL.rstrip("/") + "/", "api/tags")
DEFAULT_MODEL = "qwen3.6:27b"                     # override with --model
# ---------------------------------------------------------------------------
# Broad sectors & Canonical Examples
# ---------------------------------------------------------------------------
SECTORS = [
    "Technology & IT", "Engineering", "Healthcare & Medical", "Education & Training",
    "Finance & Accounting", "Banking & Insurance", "Sales & Business Development",
    "Marketing, Advertising & PR", "Human Resources & Recruitment", "Construction & Architecture",
    "Manufacturing & Production", "Logistics, Supply Chain & Procurement",
    "Transportation & Drivers", "Hospitality, Tourism & Food Service", "Retail & Consumer Goods",
    "Legal & Compliance", "Customer Success & Account Management", "Customer Service & Support",
    "Administration & Office Support", "Creative, Design & Arts", "Media & Communications",
    "Science & Research", "Government & Public Sector", "Skilled Trades & Maintenance",
    "Energy, Oil & Gas", "Real Estate & Property", "Consulting & Professional Services",
    "Agriculture & Environment", "Security & Safety", "Sports, Fitness & Recreation", "Other",
]
SECTORS_BULLETED = "\n".join(f"- {s}" for s in SECTORS)
_SECTOR_SET = {s.lower(): s for s in SECTORS}

ROLE_EXAMPLES = (
    "Software Engineer, Data Scientist, Data Engineer, DevOps Engineer, Cybersecurity Analyst, "
    "Network Engineer, IT Support Specialist, Systems Administrator, Cloud Engineer, "
    "Mobile Developer, Frontend Developer, Backend Developer, AI Engineer, QA Engineer; "
    "Civil Engineer, Mechanical Engineer, Electrical Engineer, Chemical Engineer, "
    "Industrial Engineer, Structural Engineer, HVAC Engineer, Site Engineer, Architect; "
    "Registered Nurse, Physician, Pharmacist, Dentist, Surgeon, Radiologist, "
    "Physiotherapist, Lab Technician, Medical Receptionist, Dietitian, Paramedic; "
    "Teacher, University Lecturer, Teaching Assistant, School Principal, Curriculum Developer, Academic Advisor; "
    "Accountant, Financial Analyst, Auditor, Tax Accountant, Bookkeeper, "
    "Bank Teller, Relationship Manager, Credit Analyst, Underwriter, Actuary, Investment Analyst; "
    "Sales Representative, Account Manager, Business Development Manager, Retail Sales Associate, "
    "Marketing Manager, Digital Marketing Specialist, Content Writer, SEO Specialist, Social Media Manager, "
    "Brand Manager, HR Specialist, Recruiter, Talent Acquisition Specialist, Payroll Specialist; "
    "Project Manager, Quantity Surveyor, Foreman, Safety Officer, "
    "Production Supervisor, Quality Inspector, Machine Operator, Maintenance Technician, "
    "Logistics Coordinator, Warehouse Supervisor, Procurement Officer, Supply Chain Analyst, "
    "Truck Driver, Delivery Driver, Fleet Manager, Pilot, Flight Attendant; "
    "Chef, Cook, Waiter, Barista, Hotel Manager, Housekeeper, Event Coordinator, Travel Consultant, "
    "Store Manager, Cashier, Merchandiser, Customer Service Representative, Call Center Agent, "
    "Retention Specialist, Customer Success Manager, Account Executive; "
    "Administrative Assistant, Office Manager, Executive Secretary, Data Entry Clerk, Receptionist, "
    "Lawyer, Legal Counsel, Compliance Officer, Paralegal, "
    "Graphic Designer, Interior Designer, UX Designer, Photographer, Video Editor, "
    "Journalist, Public Relations Officer, Copywriter, Translator, "
    "Research Scientist, Chemist, Biologist, Lab Researcher; "
    "Petroleum Engineer, Drilling Engineer, Electrical Technician, "
    "Real Estate Agent, Property Manager, Management Consultant, Business Analyst, "
    "Agronomist, Farm Manager, Environmental Specialist, "
    "Security Guard, Safety Inspector, "
    "Fitness Trainer, Sports Coach, "
    "Electrician, Plumber, Welder, Carpenter, HVAC Technician, Auto Mechanic"
)

SUMMARISER_PROMPT = """You are a job description analyst. Read the description carefully and extract the core responsibilities. Then, based ONLY on those responsibilities, generate a clean, standard job title (e.g. "Data Scientist", "Marketing Manager", "Customer Service Representative") that best matches what the person will do day-to-day.

IMPORTANT: Ignore any job title that might be given elsewhere. Use ONLY the description.

Use these well-known titles as a reference (but you are not limited to them):
{examples}

Return ONLY a JSON object with exactly these keys:
- "cleaned_title": a standard job title in Title Case (1-3 words, no seniority words like Senior/Junior/Lead)
- "duties": an array of 3-5 short bullet points summarising the main activities

Job description:
{description}

Reply with JSON only (no extra text):"""

CLASSIFIER_PROMPT = """You are an expert job classifier covering EVERY industry.

You will receive a cleaned job title and a list of core duties. Use these to determine the sector and the most specific, standard job category.

STEP 1 — Pick ONE broad sector from this list:
{sectors}

STEP 2 — Give the single most SPECIFIC, standard job category (a clean, canonical job title in Title Case, 1-3 words). Use widely-recognised real-world titles like these (you are NOT limited to them):
{examples}

RULES (strictly follow):
- Determine the sector based ONLY on the job function (what the person DOES day-to-day), NOT on the company's industry.
- The job title provided is already a cleaned, duty-based title – trust it, but cross-check with the duties.
- "category" must be a clean job title ONLY — NO seniority words, NO company names, NO locations.
- Prefer a commonly-used standard title so similar jobs get the same label.
- Every job fits somewhere — never leave it blank; use "Other" only as last resort.
- Never mention the original job posting's title (you do not know it).

Cleaned job title: {cleaned_title}
Core duties:
{duties}

Reply with ONLY this JSON (no extra text, no markdown):
{{"sector": "<one sector from the list>", "category": "<specific job title>", "certainty": "high|medium|low"}}

"certainty" guide:
- high   – The duties leave no doubt; the role is clearly this category.
- medium – Mostly matches, but some ambiguity.
- low    – You are guessing; the duties are vague or overlap multiple categories.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truncate(text: str, n: int = 12000) -> str:
    """Truncate to n chars – safe for our 16k context window."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n] + "..."


def extract_json(text: str) -> str:
    """
    Try to extract a JSON object from a model response that may contain
    markdown fences, surrounding text, or stray characters.
    """
    if not text or not text.strip():
        return ""

    # Remove markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```", "", text)

    # Find the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Async Workers
# ---------------------------------------------------------------------------
async def writer_task(queue: asyncio.Queue, out_path: Path, total_rows: int):
    """Background task to write results cleanly without lock contention."""
    counter = {"n": 0, "err": 0}
    started = time.time()
    
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            rec = await queue.get()
            if rec is None:
                queue.task_done()
                break
                
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counter["n"] += 1
            if rec.get("llm_certainty") == "ERROR":
                counter["err"] += 1
                
            if counter["n"] % 10 == 0:
                f.flush()
                
            n = counter["n"]
            if n % 20 == 0 or n == total_rows:
                el = time.time() - started
                rate = n / el if el > 0 else 0
                eta = (total_rows - n) / rate / 60 if rate else 0
                print(
                    f"[{n:>6}/{total_rows}] {rate:4.1f} rows/s  "
                    f"ETA {eta:5.1f}m  errors {counter['err']}  "
                    f"last: {rec.get('sector', 'N/A')} / {rec.get('llm_category', 'N/A')}",
                    flush=True,
                )
            queue.task_done()


async def process_row(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                      model: str, row, queue: asyncio.Queue, timeout=120):
    """Runs the two-stage classification asynchronously."""
    async with sem:
        try:
            desc = truncate(row.job_description if isinstance(row.job_description, str) else "")
            
            # ---------- Stage 1: Summarise ----------
            resp1 = await session.post(
                OLLAMA_API_GENERATE,
                json={
                    "model": model,
                    "prompt": SUMMARISER_PROMPT.format(examples=ROLE_EXAMPLES, description=desc),
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "num_ctx": 16384,        # 16k context – safe for 12k descriptions
                        "num_predict": 300,
                    },
                },
                timeout=timeout,
            )
            resp1.raise_for_status()
            res1_data = await resp1.json()
            raw1 = res1_data["response"]
            clean1 = extract_json(raw1)
            if not clean1:
                raise ValueError(f"Stage 1: no JSON found. Raw: {raw1[:200]}")
            summary = json.loads(clean1)

            cleaned_title = str(summary.get("cleaned_title", "")).strip() or "Other"
            duties = summary.get("duties", [])
            if not duties or not isinstance(duties, list):
                duties = ["No duties extracted"]
            duties_formatted = "\n".join(f"- {d}" for d in duties)

            # ---------- Stage 2: Classify ----------
            resp2 = await session.post(
                OLLAMA_API_GENERATE,
                json={
                    "model": model,
                    "prompt": CLASSIFIER_PROMPT.format(
                        sectors=SECTORS_BULLETED,
                        examples=ROLE_EXAMPLES,
                        cleaned_title=cleaned_title,
                        duties=duties_formatted,
                    ),
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "num_ctx": 16384,
                        "num_predict": 100,
                    },
                },
                timeout=timeout,
            )
            resp2.raise_for_status()
            res2_data = await resp2.json()
            raw2 = res2_data["response"]
            clean2 = extract_json(raw2)
            if not clean2:
                raise ValueError(f"Stage 2: no JSON found. Raw: {raw2[:200]}")
            parsed = json.loads(clean2)

            # Normalise
            sector_raw = str(parsed.get("sector", "")).strip()
            sector = _SECTOR_SET.get(sector_raw.lower(), "Other")
            category = str(parsed.get("category", "")).strip() or "Other"
            for prefix in ("Senior ", "Junior ", "Lead ", "Principal ", "Sr. ", "Jr. "):
                if category.startswith(prefix):
                    category = category[len(prefix):].strip()
            certainty_raw = str(parsed.get("certainty", "low")).strip().lower()
            if certainty_raw not in ("high", "medium", "low"):
                certainty_raw = "low"

            rec = {
                "job_id": row.job_id,
                "job_title": row.job_title,
                "sector": sector,
                "llm_category": category[:60],
                "llm_certainty": certainty_raw,
                "cleaned_title": cleaned_title,
                "extracted_duties": duties,
            }

        except Exception as e:
            rec = {
                "job_id": row.job_id,
                "job_title": getattr(row, 'job_title', ''),
                "sector": "Other",
                "llm_category": "ERROR",
                "llm_certainty": "ERROR",
                "cleaned_title": "",
                "extracted_duties": [],
                "error": f"{type(e).__name__}: {e}"[:200],
                # Include a snippet of the raw responses (if captured) for debugging
                "raw_stage1": raw1[:120] if 'raw1' in locals() else "",
                "raw_stage2": raw2[:120] if 'raw2' in locals() else "",
            }

        await queue.put(rec)


# ---------------------------------------------------------------------------
# Resume & CSV Logic (Sync Helpers)
# ---------------------------------------------------------------------------
def load_done(out_path: Path) -> set:
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
    cols = ["job_id", "job_title", "company", "location", "sector", "llm_category", "llm_certainty"]
    out = labels[[c for c in cols if c in labels.columns]].rename(
        columns={"llm_category": "category", "llm_certainty": "certainty"}
    )
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV written: {csv_path}  ({len(out):,} rows)")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
async def async_main(args, todo_df, out_path, csv_path):
    queue = asyncio.Queue()
    total_rows = len(todo_df)

    writer = asyncio.create_task(writer_task(queue, out_path, total_rows))

    connector = aiohttp.TCPConnector(limit=args.workers + 2)
    timeout = aiohttp.ClientTimeout(total=120)
    sem = asyncio.Semaphore(args.workers)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Check Ollama connection
        try:
            async with session.get(OLLAMA_API_TAGS, timeout=5) as r:
                r.raise_for_status()
        except Exception as e:
            sys.exit(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running?\n  {e}")

        tasks = [
            asyncio.create_task(process_row(session, sem, args.model, row, queue))
            for row in todo_df.itertuples(index=False)
        ]
        await asyncio.gather(*tasks)

    await queue.put(None)
    await writer

    csv_out = out_path.with_suffix(".csv")
    write_csv_from_jsonl(out_path, csv_path, csv_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.JOBS_CLEAN_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    csv_path, out_path = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    if args.limit:
        df = df.head(args.limit).copy()
    done = load_done(out_path)
    todo = df[~df["job_id"].isin(done)].copy()

    print(f"Model: {args.model} | workers: {args.workers}")
    print(f"Loaded {len(df):,} rows | already done {len(done):,} | to label {len(todo):,}")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    start_time = time.time()
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(async_main(args, todo, out_path, csv_path))
    el = (time.time() - start_time) / 60
    print(f"\nFinished in {el:.1f} minutes.")


if __name__ == "__main__":
    main()