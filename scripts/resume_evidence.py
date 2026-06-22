"""
Data-driven resume-evidence engine (replaces the old hand-written rules).

`extract_evidence(resume_text, skills_to_check)` returns, for each requested
skill that the resume supports, an evidence object:
    {"sentence": <verbatim resume sentence>,
     "reasoning": <why it is explicit / necessarily implied>,
     "confidence": "explicit" | "implied"}
Unsupported skills are omitted (the caller routes them to "skills to learn").

How it works (no manual per-skill rules — it grows with the vocabulary, the LLM
and the corpus):

  1. DETERMINISTIC EXPLICIT LAYER (no LLM). Every canonical skill that literally
     appears in the resume is found by exact, word-bounded matching, INCLUDING
     skills hidden inside compound phrases joined by "&", "/", "and", "or" or a
     shared head noun. "Data Analysis & Science" therefore yields BOTH
     "Data Analysis" and "Data Science"; "Data mining and warehousing" yields
     "Data mining" and "Data Warehouse". These are marked confidence="explicit".
     This layer is pure tokenisation, so it is exhaustive, instant and works even
     when the LLM is unreachable. (Fixes the bug where a compound phrase was read
     as one chunk and the second skill was dropped.)

  2. LLM IMPLIED LAYER (one cached pass, qwen2.5:32b). The model reasons in two
     ORDERED passes inside a single call: first it lists CONCRETE skills (named
     skills + the NECESSARY sub-tasks of described work — training a model implies
     ML, cleaning, feature engineering, evaluation; but NOT optional tools); then,
     having those concrete skills in context, it names the BROAD FIELD the
     combination demonstrates (e.g. ML + data prep + evaluation => Data Science).
     Fields are entailed; tools/frameworks/clouds are never inferred. Anything the
     LLM returns that is not already an explicit literal match is treated as
     "implied". (Fixes the bug where the umbrella skill was judged before its
     sub-skills were established.)

  3. CORROBORATION (corpus, additive only). Implied skills get a note when they
     reliably co-occur with one of the candidate's explicit skills across our
     4,885 job descriptions. The corpus is NEVER used to promote a skill on its
     own: co-occurrence is symmetric and cannot tell an umbrella field ("Data
     Science") from a co-required tool ("TensorFlow"), so promotion is left to the
     LLM's field-vs-tool judgement and the note is purely informational.

Every LLM-quoted sentence is matched back to a real resume sentence (token
Jaccard >= .34); unmatched quotes are dropped, so nothing is fabricated. On LLM
failure the deterministic explicit layer still returns, and everything else
routes to the static learning resources.

The expensive LLM pass is memoised per resume, so cost is paid once per resume
(the Streamlit app additionally caches the result in st.session_state).
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.extract_skills import _patterns
from scripts.skill_vocab import SKILL_ALIASES, SKILL_SYNONYMS, SKILLS_VOCABULARY

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
COOC_CACHE = PROJECT_ROOT / "data" / "processed" / "skill_cooc.json"
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

# Map any alias/synonym surface form to one lowercase canonical key, so that
# "LLMs", "Natural Language Processing", etc. match the skills the caller asks
# about regardless of phrasing.
_CANON = {k.lower(): v.lower() for k, v in {**SKILL_ALIASES, **SKILL_SYNONYMS}.items()}


def _norm(skill: str) -> str:
    s = re.sub(r"\s+", " ", str(skill).strip().lower()).strip(" .,:;")
    return _CANON.get(s, s)


def _sentences(text: str) -> list[str]:
    """Resume lines/bullets as evidence units (split on newlines/bullets, not
    on '.' — periods inside emails, URLs and abbreviations must not break)."""
    parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"[\n•]+", text or "")]
    return [p for p in parts if len(p) >= 12]


# --------------------------------------------------------------------------
# Deterministic explicit layer (tokenisation only, no LLM)
# --------------------------------------------------------------------------
# Hard segment boundaries: a skill never spans these. Soft coordinators inside a
# segment ("&", "/", "and", "or", "+") may share a head/tail noun across items.
_HARD = re.compile(r"[,;:()\[\]–—]|\s-\s")
_COORD = re.compile(r"\s*&\s*|\s+and\s+|\s+or\s+|\s+/\s+|\s+\+\s+", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#/-]*")


def _normsurf(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


@functools.lru_cache(maxsize=1)
def _surface_map() -> dict:
    """{normalised surface form -> (canonical key, display name)} for every vocab
    skill and every known alias/synonym spelling. Used to resolve the distributed
    candidates below to real skills (exact match — never fuzzy)."""
    m: dict[str, tuple[str, str]] = {}
    for s in SKILLS_VOCABULARY:
        m[_normsurf(s)] = (_norm(s), s)
    for variant, canon in {**SKILL_ALIASES, **SKILL_SYNONYMS}.items():
        m.setdefault(_normsurf(variant), (_norm(canon), canon))
    return m


def _distributed_candidates(sentence: str):
    """Yield (candidate_surface, segment_text) for compound phrases where a head
    or tail noun is shared across a conjunction. "Data Analysis & Science" yields
    the candidate "data science" (head "Data" shared with "Science"); the segment
    is returned as the focused evidence span. Candidates are only *proposed* here
    — they become evidence only if they exactly match a real skill."""
    for seg in _HARD.split(sentence):
        seg = seg.strip()
        if not seg or _COORD.search(seg) is None:
            continue
        frags = [_WORD.findall(f) for f in _COORD.split(seg)]
        frags = [f for f in frags if f]
        if len(frags) < 2:
            continue
        cands: set[tuple] = set()
        head = frags[0]                       # head-shared: "Data [Analysis & Science]"
        for f in frags[1:]:
            for i in range(1, len(head)):
                cands.add(tuple(head[:i] + f))
        tail = frags[-1]                      # tail-shared: "[model & data] Engineering"
        for f in frags[:-1]:
            for j in range(1, len(tail)):
                cands.add(tuple(f + tail[-j:]))
        for c in cands:
            yield " ".join(c).lower(), seg


def _literal_explicit(sentences: list[str]) -> dict:
    """Every canonical skill literally present in the resume -> (display name,
    evidence span). Direct word-bounded matches use the shared vocabulary
    patterns; compound phrases additionally use distributive expansion so the
    second skill behind "&"/"/"/"and" is never missed."""
    pats = _patterns()
    smap = _surface_map()
    out: dict[str, tuple[str, str]] = {}
    for sent in sentences:
        for canon, pat in pats:               # direct matches (evidence = full line)
            if pat.search(sent):
                out.setdefault(_norm(canon), (canon, sent))
        for cand, seg in _distributed_candidates(sent):   # compound matches
            hit = smap.get(cand)
            if hit:
                out.setdefault(hit[0], (hit[1], seg))      # evidence = focused phrase
    return out


# --------------------------------------------------------------------------
# LLM implied layer (one cached pass, ordered concrete -> fields)
# --------------------------------------------------------------------------
PROMPT = """You are an expert technical recruiter. List every technical SKILL the \
resume demonstrates, with traceable evidence, working in TWO ORDERED passes.

PASS 1 - CONCRETE skills. Go through EVERY experience bullet, project, \
certification and coursework item. For each, output:
- every skill/tool/technology NAMED in that text, and
- every skill that is a NECESSARY sub-task of the described work (the work is \
impossible or implausible without it). Be exhaustive and apply fully:
  * Training, benchmarking or deploying ANY ML/AI model -> you MUST output ALL \
FOUR of these as separate skills EVERY time, each with the same quoting sentence, \
never fewer: Machine Learning, Data Cleaning, Feature Engineering, Model \
Evaluation. Additionally add Deep Learning if a neural network / Transformer / \
deep model is mentioned, and NLP if the work is over text/language.
  * Building/structuring a dataset from raw data -> Data Cleaning and Data Engineering.
  * Each "Relevant Coursework" subject -> its core concept (a Project Management \
course -> Agile; a Database course -> SQL and Data Modeling).
Do NOT infer OPTIONAL tools the work could be done without unless they are NAMED \
(Weights & Biases, TensorBoard, MLflow, clouds AWS/Azure/GCP, GPUs). If such a \
tool IS named, include it.

PASS 2 - BROAD FIELDS. Now look at the concrete skills you listed in PASS 1 \
TOGETHER, and output the broad FIELD(s) that this COMBINATION necessarily \
demonstrates. A field is an umbrella discipline, NOT a tool. For example, \
machine learning + data preparation + model evaluation together demonstrate \
Data Science. Only output a field when several of its concrete sub-skills are \
already present in PASS 1. NEVER output a tool, framework, library or cloud as a \
field.

Every skill MUST quote a verbatim sentence from the resume (for a PASS 2 field, \
quote the strongest sentence behind its sub-skills). Never invent a skill, \
sentence, or experience the resume does not support.

Resume:
<<<RESUME>>>

Return ONLY a JSON object:
{"skills": [{"skill": "<skill>", "sentence": "<verbatim resume sentence>", \
"reasoning": "<one sentence: why named or necessarily implied>", \
"confidence": "explicit|implied"}]}"""


def _parse_skills(raw: str) -> list[dict]:
    """Parse the model's {"skills":[...]} — tolerant of a truncated array by
    salvaging every complete flat object that has a "skill" field."""
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and isinstance(d.get("skills"), list):
            return [s for s in d["skills"] if isinstance(s, dict) and s.get("skill")]
    except Exception:
        pass
    out = []
    for mobj in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            o = json.loads(mobj.group())
            if isinstance(o, dict) and o.get("skill"):
                out.append(o)
        except Exception:
            pass
    return out


def _call_llm(resume: str) -> list[dict]:
    prompt = PROMPT.replace("<<<RESUME>>>", resume[:6000] or "(empty)")
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "format": "json", "stream": False,
              "keep_alive": "15m",
              "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 2048}},
        timeout=180)
    resp.raise_for_status()
    return _parse_skills(resp.json()["response"])


def _best_sentence(quoted: str, sentences: list[str]) -> str | None:
    """Return the real resume sentence most overlapping `quoted` (token Jaccard),
    or None if nothing meaningfully matches — guards against invented quotes."""
    q = set(re.findall(r"[a-z0-9]+", (quoted or "").lower()))
    if not q:
        return None
    best, score = None, 0.0
    for s in sentences:
        toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        if not toks:
            continue
        j = len(q & toks) / len(q | toks)
        if j > score:
            best, score = s, j
    return best if score >= 0.34 else None


# --------------------------------------------------------------------------
# Corpus co-occurrence (data-driven CORROBORATION only — never promotion)
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _cooc() -> dict:
    """Load (or build + cache) skill co-occurrence over the job-description
    corpus: {df: {skill:#postings}, co: {"a|b":#postings}, n: total}."""
    if COOC_CACHE.exists():
        try:
            return json.loads(COOC_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        import pandas as pd
        from scripts.extract_skills import extract_skills
        df_counts, co = {}, {}
        descs = pd.read_csv(CLEANED_CSV)["clean_description"].dropna().tolist()
        for d in descs:
            present = sorted({s.lower() for s in extract_skills(d)})
            for s in present:
                df_counts[s] = df_counts.get(s, 0) + 1
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    co[f"{a}|{b}"] = co.get(f"{a}|{b}", 0) + 1
        out = {"df": df_counts, "co": co, "n": len(descs)}
        COOC_CACHE.write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception:
        return {"df": {}, "co": {}, "n": 0}


def _corpus_note(skill: str, explicit: list[str]) -> str:
    """If `skill` reliably co-occurs with one of the candidate's explicit skills
    in real postings, return a short corroborating note (else ''). Additive only:
    this never decides whether a skill is supported, only annotates one the LLM
    already inferred."""
    c = _cooc()
    df, co = c.get("df", {}), c.get("co", {})
    a = _norm(skill)
    if df.get(a, 0) < 5:
        return ""
    best_e, best_p = None, 0.0
    for e in explicit:
        b = _norm(e)
        if b == a or df.get(b, 0) < 5:
            continue
        pair = "|".join(sorted((a, b)))
        p = co.get(pair, 0) / df[b]
        if p > best_p:
            best_e, best_p = e, p
    if best_e and best_p >= 0.25:
        return (f" Corpus check: appears with {best_e} in {best_p:.0%} of "
                f"job postings that mention {best_e}.")
    return ""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _demonstrated_index(resume_text: str) -> tuple:
    """Deterministic explicit layer + LLM implied layer + corpus corroboration,
    merged (explicit always wins). Memoised per resume. Returns a tuple of
    (canonical_key, display_name, evidence_dict)."""
    sentences = _sentences(resume_text)

    # 1. Deterministic explicit — authoritative for everything literally present.
    literal = _literal_explicit(sentences)
    explicit_names = [disp for disp, _sent in literal.values()]
    index: dict[str, tuple[str, dict]] = {}
    for key, (disp, sent) in literal.items():
        index[key] = (disp, {"sentence": sent,
                             "reasoning": "Stated explicitly in your resume.",
                             "confidence": "explicit"})

    # 2. LLM-inferred — necessary sub-tasks + broad fields. Anything not already
    #    an explicit literal match is recorded as "implied" (the LLM no longer
    #    decides what counts as explicit — tokenisation does).
    try:
        items = _call_llm(resume_text)
    except Exception:
        items = []
    for it in items:
        name = it.get("skill")
        if not name:
            continue
        key = _norm(name)
        if key in index:                       # already explicit -> keep it
            continue
        sent = _best_sentence(it.get("sentence", ""), sentences)
        if not sent:                           # quote not in resume -> drop
            continue
        reasoning = re.sub(r"\s+", " ", str(it.get("reasoning", ""))).strip()
        reasoning += _corpus_note(name, explicit_names)
        index[key] = (name, {"sentence": sent, "reasoning": reasoning,
                             "confidence": "implied"})

    return tuple((key, disp, ev) for key, (disp, ev) in index.items())


def extract_evidence(resume_text: str, skills_to_check) -> dict:
    """Evidence for each requested skill the resume supports (see module doc).

    Skills with no support are omitted. Safe if the LLM is unreachable (the
    deterministic explicit layer still returns; everything else routes to the
    static learning resources)."""
    try:
        index = _demonstrated_index(resume_text or "")
    except Exception:
        return {}
    by_key = {}
    for key, _name, ev in index:
        by_key.setdefault(key, ev)             # first (explicit precedes implied) wins
    result = {}
    for skill in skills_to_check or []:
        ev = by_key.get(_norm(skill))
        if ev:
            result[skill] = ev
    return result
