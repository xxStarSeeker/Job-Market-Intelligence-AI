"""
Streamlit demo for TalentFit.

Candidate-facing UI: upload a resume, set a profile, get job matches
(embedding "Fast Search" or skill-aware "Best Match"), plus an optional
admin tool that classifies a job description with the tech-11 ensemble.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_skills import extract_skills
from scripts.predict import predict_category
from scripts.recommend import get_available_roles
from scripts.resume_evidence import extract_evidence
from scripts.skill_resources import resource_for
from scripts.skill_vocab import get_skill_vocabulary

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

st.set_page_config(page_title="TalentFit", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; color: #0f172a; }
    .block-container { padding-top: 1.5rem; max-width: 1180px; }
    .hero {
        background: #ffffff;
        border: 1px solid #d9e2ef;
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 4px 0; font-size: 2.2rem; letter-spacing: 0; }
    .hero p { margin: 0; color: #475569; font-size: 1rem; }
    .section-label {
        color: #2563eb;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-top: 12px;
    }
    .section-title { font-size: 1.25rem; font-weight: 750; margin-bottom: 8px; }
    .job-card {
        background: #ffffff;
        border: 1px solid #d9e2ef;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .job-title { font-size: 1.15rem; font-weight: 750; margin-bottom: 6px; }
    .job-meta { color: #526174; font-size: 0.92rem; margin-bottom: 14px; }
    .score-pill {
        float: right;
        background: #dcfce7;
        color: #047857;
        border-radius: 999px;
        padding: 8px 16px;
        font-weight: 750;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        margin: 3px 5px 3px 0;
        font-size: 0.86rem;
        font-weight: 650;
    }
    .pill-ok { background: #e0f2fe; color: #075985; }
    .pill-gap { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
    .muted { color: #64748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def job_lookup() -> pd.DataFrame:
    """job_id -> display metadata for result cards and tables."""
    df = pd.read_csv(
        PROJECT_ROOT / "job_data_saudi_clean.csv",
        usecols=["job_id", "job_title", "job_description", "company", "location", "url"],
    )
    return df.drop_duplicates("job_id").set_index("job_id")


def extract_resume(uploaded) -> str:
    if uploaded.name.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            st.warning("Install PyPDF2 for PDF support. Paste text below instead.")
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(uploaded).pages)
    return uploaded.read().decode("utf-8", errors="replace")


def extract_experience_hint(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"\b\d{1,2}\s*\+?\s*(?:years?|yrs?)\b", text.lower())
    if matches:
        return ", ".join(matches[:2])
    entry_patterns = (
        r"\brecent(?:ly)?\b.*\b(graduat(?:e|ed|ing))\b",
        r"\bfresh\s+graduate\b",
        r"\bnew\s+graduate\b",
        r"\bgraduate\s+trainee\b",
        r"\bgraduated\b(?!\s+(degree|studies|school|program|certificate|diploma))",
        r"\bgraduate\b(?!\s+(degree|studies|school|program|certificate|diploma))",
        r"\bentry[-\s]?level\b",
        r"\bjunior\s+(candidate|role|position|developer|analyst|engineer|specialist)\b",
        r"\bintern(ship)?\b",
        r"\btrainee\b",
        r"\bno\s+(prior\s+)?experience\b",
    )
    for line in text.splitlines():
        if any(re.search(pattern, line.lower()) for pattern in entry_patterns):
            return "entry-level / recent graduate"
    return ""


def score_color(score: float) -> str:
    if score >= 0.5:
        return "background-color: #dcfce7; color: #047857"
    return "background-color: #fef3c7; color: #92400e" if score >= 0.3 else "background-color: #e5e7eb; color: #374151"


def score_badge(score: float) -> str:
    return f"<span class='score-pill'>{score:.3f}</span>"


def safe_text(value, default: str = "?") -> str:
    return default if pd.isna(value) or value in ("", None) else str(value)


def results_frame(matches: list[dict]) -> pd.DataFrame:
    look = job_lookup()
    rows = []
    for m in matches:
        row = look.loc[m["job_id"]] if m["job_id"] in look.index else {}
        rows.append({
            "Job Title": safe_text(row.get("job_title") if isinstance(row, pd.Series) else None),
            "Company": safe_text(row.get("company") if isinstance(row, pd.Series) else None),
            "Location": safe_text(row.get("location") if isinstance(row, pd.Series) else None),
            "Category": m["category"],
            "Match Score": m["score"],
            "Reason": m["reason"],
            "Apply URL": safe_text(row.get("url") if isinstance(row, pd.Series) else None, ""),
            "Job Description": safe_text(row.get("job_description") if isinstance(row, pd.Series) else None, ""),
        })
    return pd.DataFrame(rows)


def _resume_evidence() -> dict:
    """Evidence index for the current resume, computed once per resume."""
    resume = st.session_state.get("resume_text", "")
    if st.session_state.get("_evidence_src") != resume:
        with st.spinner("Analyzing your resume for evidence (once per resume)..."):
            st.session_state["resume_evidence"] = extract_evidence(
                resume, get_skill_vocabulary()
            )
        st.session_state["_evidence_src"] = resume
    return st.session_state.get("resume_evidence", {})


def optimize_resume(job: dict, title: str) -> str:
    """Evidence-grounded, two-part optimization for one job."""
    resume = st.session_state.get("resume_text", "")
    listed = {s.lower() for s in st.session_state.get("skills", [])}
    missing = [s for s in job.get("missing_skills", []) if s.lower() not in listed]
    evidence = _resume_evidence()
    evidence_lower = {k.lower(): (k, v) for k, v in evidence.items()}
    supported_gaps = [
        (s, evidence_lower[s.lower()][1])
        for s in missing
        if s.lower() in evidence_lower
    ]
    true_gaps = [s for s in missing if s.lower() not in evidence_lower]
    return (
        _quick_wins_md(job, title, resume, supported_gaps)
        + "\n\n"
        + _learn_md(true_gaps)
    )


def _quick_wins_md(job: dict, title: str, resume: str, supported_gaps: list) -> str:
    """Skills the resume already proves, with optional LLM rewording."""
    head = "### ✅ Skills you already have — add these (backed by your resume)"
    if not supported_gaps:
        return (
            head
            + "\n\n"
            + "Nothing to add: your listed skills already cover the evidence in your resume for this job."
        )

    add_line = "**Add to your Skills section:** " + ", ".join(
        skill for skill, _ in supported_gaps
    )
    evidence = "\n".join(
        f'- **{skill}** ({ev["confidence"]}) — "{ev["sentence"]}" -> {ev["reasoning"]}'
        for skill, ev in supported_gaps
    )
    facts = "\n".join(
        f'{skill} | evidence: "{ev["sentence"]}" | {ev["reasoning"]}'
        for skill, ev in supported_gaps
    )
    prompt = "\n".join([
        f"You are a precise resume editor for a '{title}' ({job['category']}) application.",
        "Plain professional language; no buzzwords (leverage, robust, spearheaded, state-of-the-art).",
        "Each skill below already has real evidence in the resume. For EACH, write ONE",
        "concrete suggestion: how to surface it (a Skills-section entry and/or a bullet",
        "rephrasing) using ONLY its quoted evidence. Invent nothing; add no other skill.",
        "",
        facts,
        "",
        "Return a short markdown bullet list, one bullet per skill.",
    ])
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OPT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 500},
            },
            timeout=60,
        )
        response.raise_for_status()
        rewording = "**How to surface them:**\n" + response.json()["response"].strip()
    except Exception as exc:
        rewording = (
            f"_(AI rewording unavailable: {type(exc).__name__}; "
            "the evidence above is what to add.)_"
        )
    return head + "\n\n" + add_line + "\n\n" + evidence + "\n\n" + rewording


def _learn_md(true_gaps: list) -> str:
    """Genuine gaps, each with one static free resource."""
    head = "### 📚 Skills to learn (free resources)"
    if not true_gaps:
        return (
            head
            + "\n\n"
            + "No remaining gaps for this job — you have evidence for everything it asks."
        )
    return head + "\n\n" + "\n".join(
        f"- **{skill}** — {resource_for(skill)}" for skill in true_gaps
    )


def render_gap(job: dict, title: str, company: str, key: str) -> None:
    st.write(f"**{title} - {company}**")
    st.write("**Matched skills:** " + (", ".join(job.get("matched_skills", [])) or "none"))
    st.write("**Missing skills:** " + (", ".join(job.get("missing_skills", [])) or "none"))
    if st.button("Optimize Resume for This Job", key=f"opt_{key}"):
        with st.spinner("Optimizing resume with AI..."):
            suggestion = optimize_resume(job, title)
            current = dict(st.session_state.get("opt", {}))
            current[job["job_id"]] = suggestion
            st.session_state["opt"] = current
    suggestion = st.session_state.get("opt", {}).get(job["job_id"])
    if suggestion:
        st.markdown("**Resume suggestions**")
        st.markdown(suggestion)


def pill_list(values: list[str], kind: str) -> str:
    if not values:
        return "<em>None listed</em>"
    cls = "pill-ok" if kind == "ok" else "pill-gap"
    return "".join(f"<span class='pill {cls}'>{v}</span>" for v in values[:8])


def render_job_card(job: dict, row: pd.Series, rank: int) -> None:
    title = safe_text(row["Job Title"])
    company = safe_text(row["Company"])
    url = safe_text(row["Apply URL"], "")
    desc = safe_text(row["Job Description"], "")
    st.markdown(
        f"""
        <div class="job-card">
            {score_badge(job["score"])}
            <div class="job-title">{rank}. {title}</div>
            <div class="job-meta">{company} | {safe_text(row["Location"])} | {safe_text(row["Category"])}</div>
            <div class="muted">{safe_text(row["Reason"], "")}</div>
            <br>
            <strong>MATCHED SKILLS</strong><br>{pill_list(job.get("matched_skills", []), "ok")}
            <br><br>
            <strong>SKILL GAPS</strong><br>{pill_list(job.get("missing_skills", []), "gap")}
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    if url:
        c1.link_button("Apply", url, width="stretch")
    else:
        c1.button("Apply", disabled=True, width="stretch", key=f"apply_disabled_{rank}")
    if c2.button("View Job Description", key=f"desc_{rank}", width="stretch"):
        st.session_state[f"show_desc_{rank}"] = not st.session_state.get(f"show_desc_{rank}", False)
    if c3.button("Resume Tips", key=f"tips_{rank}", width="stretch"):
        st.session_state[f"show_tips_{rank}"] = not st.session_state.get(f"show_tips_{rank}", False)
    if st.session_state.get(f"show_desc_{rank}"):
        with st.expander(f"Job Description - {title}", expanded=True):
            st.write(desc or "No description available.")
    if st.session_state.get(f"show_tips_{rank}"):
        with st.expander(f"Resume Tips - {title}", expanded=True):
            render_gap(job, title, company, key=f"card_{rank}")


def render_empty_match_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="job-card">
            <div class="job-title">{title}</div>
            <div class="muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30)
def backend_ok() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


st.markdown(
    """
    <div class="hero">
      <h1>TalentFit</h1>
      <p>Match a candidate resume to relevant Saudi job postings with skills, experience, and resume guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("System")
    if backend_ok():
        st.success("Backend connected")
    else:
        st.error("Backend is not reachable. Start FastAPI first.")
    st.caption(API_URL)

st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Resume</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Resume (.txt or .pdf)", type=["txt", "pdf"])

if uploaded is not None:
    upload_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("_upload_key") != upload_key:
        st.session_state["_upload_key"] = upload_key
        for key in ("opt", "_evidence_src", "resume_evidence"):
            st.session_state.pop(key, None)
        with st.spinner("Reading resume..."):
            st.session_state["resume_text"] = extract_resume(uploaded)
            st.session_state["skills"] = extract_skills(st.session_state["resume_text"])
            detected = extract_experience_hint(st.session_state["resume_text"])
            if detected and not st.session_state.get("experience"):
                st.session_state["experience"] = detected
        st.success("Resume loaded")

if st.button("Clear Resume & Search"):
    for key in (
        "resume_text", "results", "strategy_used", "opt", "_upload_key",
        "_evidence_src", "resume_evidence"
    ):
        st.session_state.pop(key, None)
    st.session_state["skills"] = []
    st.session_state["experience"] = ""
    st.rerun()

if st.session_state.get("resume_text"):
    with st.expander("Extracted Resume Text"):
        st.text(st.session_state["resume_text"][:5000])

st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Candidate Profile</div>', unsafe_allow_html=True)

roles = get_available_roles()
skill_vocab = get_skill_vocabulary()
st.session_state.setdefault("target_roles", [])
st.session_state.setdefault("skills", [])
st.session_state.setdefault("experience", "")

st.multiselect(
    "Target Roles",
    roles,
    key="target_roles",
    help="Optional: select target roles to narrow the search.",
)
st.multiselect(
    "Skills",
    skill_vocab,
    key="skills",
    help="Upload a resume to pre-fill this list, then adjust it.",
)
if st.session_state.get("resume_text"):
    st.caption(f"Detected {len(st.session_state['skills'])} skills from the resume. Edit above to add or remove.")
st.text_area(
    "Experience",
    key="experience",
    height=96,
    placeholder="Example: 2 years as a junior data analyst, SQL dashboards, Python reporting",
)
st.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.05, key="min_score")

col_fast, col_best = st.columns(2)
fast = col_fast.button("Fast Search", width="stretch", disabled=not backend_ok())
best = col_best.button("Best Match", width="stretch", type="primary", disabled=not backend_ok())

if fast or best:
    skills = st.session_state["skills"]
    resume_text = st.session_state.get("resume_text", "")
    experience = (
        st.session_state["experience"].strip()
        or extract_experience_hint(resume_text)
        or resume_text[:1500]
    )
    if not skills and not experience:
        st.error("Enter at least your skills or upload a resume.")
    else:
        strategy = "embedding" if fast else "hybrid"
        payload = {
            "skills": skills,
            "target_roles": st.session_state["target_roles"],
            "experience": experience,
            "min_score": st.session_state["min_score"],
            "strategy": strategy,
        }
        status = st.status("Searching jobs..." if fast else "Finding best matches...", expanded=True)
        status.write("Preparing candidate profile")
        status.write("Matching against job descriptions")
        if best:
            status.write("Balancing semantic match with skill coverage")
        try:
            response = requests.post(f"{API_URL}/recommend", json=payload, timeout=90)
            if response.status_code == 200:
                st.session_state["results"] = response.json()
                st.session_state["strategy_used"] = strategy
                status.update(label="Matches ready", state="complete", expanded=False)
            else:
                status.update(label="Search failed", state="error", expanded=False)
                st.error(f"API Error: {response.status_code} - {response.text}")
        except requests.RequestException:
            status.update(label="Backend is not reachable", state="error", expanded=False)
            st.error("Start FastAPI first, then run Streamlit.")

if st.session_state.get("results") is not None:
    raw_results = st.session_state["results"]
    if st.session_state.get("strategy_used") == "hybrid":
        results = [job for job in raw_results if job.get("matched_skills")]
    else:
        results = raw_results
    st.markdown('<div class="section-label">Step 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Jobs</div>', unsafe_allow_html=True)
    st.metric("Matches Found", len(results))
    if results:
        display_results = results[:50]
        df = results_frame(display_results)
        st.subheader("Top Matches")
        for i, job in enumerate(display_results[:5]):
            render_job_card(job, df.iloc[i], i + 1)

        with st.expander("Top Match Skill Gap & Resume Tips", expanded=True):
            render_gap(display_results[0], df.iloc[0]["Job Title"], df.iloc[0]["Company"], key="top")

        remaining = df.iloc[5:].copy()
        if not remaining.empty:
            st.subheader("More Matches")
            table = remaining.drop(columns=["Job Description"], errors="ignore")
            st.dataframe(
                table,
                column_config={"Apply URL": st.column_config.LinkColumn("Apply")},
                width="stretch",
                hide_index=True,
            )
        if len(results) > 50:
            st.caption(f"Showing top 50 of {len(results)} matches. Narrow your skills or target roles to refine results.")
        if st.session_state.get("strategy_used") == "hybrid":
            st.info("Best Match combines semantic relevance with skill coverage.")
    else:
        if st.session_state.get("strategy_used") == "hybrid":
            render_empty_match_card(
                "No matches found",
                "No jobs matched the selected skills. Add more accurate skills from the resume or choose a closer target role.",
            )
        else:
            render_empty_match_card(
                "No matches found",
                "No jobs matched the selected skills. Add more profile skills or choose a closer target role.",
            )

if st.sidebar.checkbox("Show Admin Tools"):
    st.sidebar.subheader("Job Description Classifier")
    desc = st.sidebar.text_area("Job Description", height=160)
    if st.sidebar.button("Classify") and desc.strip():
        with st.spinner("Classifying via API..."):
            payload = {"description": desc}
            try:
                response = requests.post(f"{API_URL}/classify", json=payload, timeout=60)
                if response.status_code == 200:
                    res = response.json()
                    st.sidebar.metric(res["predicted_category"], f"{res['confidence']:.1%}")
                    st.sidebar.bar_chart(pd.Series(res["all_scores"], name="probability"))
                else:
                    st.sidebar.error(f"API Error: {response.status_code}")
            except requests.RequestException:
                st.sidebar.error("Backend is not reachable. Start FastAPI first.")
