"""
Whole-word skill extraction against SKILLS_VOCABULARY.

extract_skills(text) returns canonical vocabulary skills found in the text.
It also matches known alias/synonym spellings (SKILL_ALIASES, SKILL_SYNONYMS)
and maps them to their canonical form, so "Google Cloud Platform" -> "GCP".
Matching is case-insensitive and word-bounded via (?<!\\w)/(?!\\w)
lookarounds, which behave correctly for terms containing non-word
characters (ci/cd, ai/ml, primavera p6).

    from scripts.extract_skills import extract_skills
    extract_skills("5 years Python, SQL and AWS")  # ['AWS', 'python', 'SQL']
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.skill_vocab import SKILL_ALIASES, SKILL_SYNONYMS, SKILLS_VOCABULARY

CONTEXT_SKILL_PATTERNS = {
    "Access": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)microsoft\s+access(?!\w)",
            r"(?<!\w)ms\s+access(?!\w)",
            r"(?<!\w)access\s+database(?!\w)",
        )
    ],
    "Airflow": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)apache\s+airflow(?!\w)",
            r"(?<!\w)airflow\s+(dag|dags|pipeline|pipelines|etl|scheduler)(?!\w)",
        )
    ],
    "Apache Airflow": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)apache\s+airflow(?!\w)",
            r"(?<!\w)airflow\s+(dag|dags|pipeline|pipelines|etl|scheduler)(?!\w)",
        )
    ],
    "Apache Spark": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)apache\s+spark(?!\w)",
            r"(?<!\w)spark\s+(sql|streaming|jobs?|clusters?|scala|etl|big\s+data)(?!\w)",
            r"(?<!\w)(pyspark|spark-submit)(?!\w)",
        )
    ],
    "Chef": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)chef\s+(automate|infra|cookbook|cookbooks|recipe|recipes|devops|configuration)(?!\w)",
            r"(?<!\w)(configuration\s+management|infrastructure\s+automation)\s+with\s+chef(?!\w)",
        )
    ],
    "Excel": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)microsoft\s+excel(?!\w)",
            r"(?<!\w)ms\s+excel(?!\w)",
            r"(?<!\w)excel\s+(spreadsheet|spreadsheets|pivot|pivots|vlookup|macros?|formulas?)(?!\w)",
        )
    ],
    "Go": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)golang(?!\w)",
            r"(?<!\w)go\s+(programming|language|developer|engineer|backend|microservices)(?!\w)",
        )
    ],
    "R": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)r\s+(programming|language|studio|statistical|analytics?|data\s+analysis)(?!\w)",
            r"(?<!\w)rstudio(?!\w)",
            r"(?<!\w)(tidyverse|dplyr|ggplot2)(?!\w)",
        )
    ],
    "Spark": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)apache\s+spark(?!\w)",
            r"(?<!\w)spark\s+(sql|streaming|jobs?|clusters?|pyspark|scala|etl|big\s+data)(?!\w)",
            r"(?<!\w)pyspark(?!\w)",
        )
    ],
    "Storage": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)(cloud|data|object|block|file|enterprise|distributed)\s+storage(?!\w)",
            r"(?<!\w)storage\s+(systems?|architecture|solutions?|administration|management)(?!\w)",
        )
    ],
    "Switches": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)(network|cisco|ethernet|lan)\s+switches(?!\w)",
            r"(?<!\w)switches\s+(configuration|troubleshooting|vlans?|routing)(?!\w)",
        )
    ],
    "Whisper": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)openai\s+whisper(?!\w)",
            r"(?<!\w)whisper\s+(asr|speech|transcription|speech-to-text)(?!\w)",
        )
    ],
    "Windows": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)microsoft\s+windows(?!\w)",
            r"(?<!\w)windows\s+(server|system|systems|os|administration|admin|environment|desktop)(?!\w)",
            r"(?<!\w)(active\s+directory|group\s+policy|powershell|desktop\s+support)(?!\w)",
        )
    ],
    "Word": [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?<!\w)microsoft\s+word(?!\w)",
            r"(?<!\w)ms\s+word(?!\w)",
            r"(?<!\w)word\s+processing(?!\w)",
        )
    ],
}


@functools.lru_cache(maxsize=None)
def _patterns() -> list[tuple[str, re.Pattern]]:
    """(canonical skill, whole-word pattern) for every vocab term plus every
    alias/synonym variant whose canonical form is in the vocabulary."""
    vocab = set(SKILLS_VOCABULARY)
    surfaces = [(s, s) for s in SKILLS_VOCABULARY]
    for variant, canon in {**SKILL_ALIASES, **SKILL_SYNONYMS}.items():
        if canon in vocab:
            surfaces.append((variant, canon))
    pats = []
    for surface, canon in surfaces:
        esc = re.escape(surface).replace(r"\ ", r"\s+")   # flexible inner spaces
        pats.append((canon, re.compile(rf"(?<!\w){esc}(?!\w)", re.IGNORECASE)))
    return pats


def skill_present(skill: str, text: str) -> bool:
    """True when `skill` is present in `text`.

    Some skill names are common English words in non-technical contexts. Those
    skills require stronger context so examples like "cleaned windows" do not
    become a Windows IT skill.
    """
    if not isinstance(text, str) or not text:
        return False
    if skill in CONTEXT_SKILL_PATTERNS:
        return any(pat.search(text) for pat in CONTEXT_SKILL_PATTERNS[skill])
    return any(canon == skill and pat.search(text) for canon, pat in _patterns())


def extract_skills(text: str) -> list[str]:
    """Canonical SKILLS_VOCABULARY entries in `text` (matched directly or via a
    known alias/synonym spelling), deduped and returned in vocabulary order."""
    if not isinstance(text, str) or not text:
        return []
    found = set()
    for canon, pat in _patterns():
        if canon in CONTEXT_SKILL_PATTERNS:
            if skill_present(canon, text):
                found.add(canon)
        elif pat.search(text):
            found.add(canon)
    return [s for s in SKILLS_VOCABULARY if s in found]
