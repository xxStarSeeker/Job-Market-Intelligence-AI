"""
Job recommendation with two matching strategies + benchmark.

A) match_embedding - semantic embedding search between the candidate profile
   and pre-computed job embeddings.
B) skill aware _hybrid - Fast Search candidates plus skill and experience scoring.

Embeddings: BAAI/bge-base-en-v1.5 (fallback all-MiniLM-L6-v2), CPU is fine,
cached to .npy on first use.
"""

from __future__ import annotations

import functools, re, sys  # noqa: E401
from pathlib import Path

import numpy as np, pandas as pd  # noqa: E401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.extract_skills import extract_skills, skill_present  # noqa: E402

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
PRIMARY_MODEL = "BAAI/bge-base-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HYBRID_RETRIEVAL_POOL = 100
MIN_RECOMMENDATION_SCORE = 0.30
GUARDRAIL_POOL_SIZE = 50
SEARCH_RETRIEVAL_POOL = 600
MIN_GUARDRAIL_SIMILARITY = 0.25
SEED = 42

_QUERY_STOPWORDS = {
    "about", "above", "after", "again", "also", "and", "are", "because", "been",
    "candidate", "can", "could", "cv", "experience", "experienced", "for", "from",
    "have", "has", "his", "her", "job", "more", "resume", "role", "skills",
    "that", "the", "their", "this", "with", "work", "worked", "year", "years",
    "yrs", "your",
}


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


@functools.lru_cache(maxsize=None)
def _encoder():
    """(SentenceTransformer, is_bge) - primary model with graceful fallback."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer(PRIMARY_MODEL, device=device), True
    except Exception as exc:
        _warn(f"could not load {PRIMARY_MODEL} ({exc!r}) - using {FALLBACK_MODEL}")
        return SentenceTransformer(FALLBACK_MODEL, device=device), False


@functools.lru_cache(maxsize=None)
def _try_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        _warn("faiss not installed - using NumPy brute-force semantic search")
        return None


def _faiss_index(emb: np.ndarray, index_path: Path):
    faiss = _try_faiss()
    if faiss is None:
        return None
    if index_path.exists():
        return faiss.read_index(str(index_path))
    index = faiss.IndexFlatIP(int(emb.shape[1]))
    index.add(np.ascontiguousarray(emb, dtype="float32"))
    faiss.write_index(index, str(index_path))
    return index


@functools.lru_cache(maxsize=None)
def _data() -> tuple[pd.DataFrame, np.ndarray, object]:
    """Jobs DataFrame + normalized doc embeddings + FAISS index."""
    df = pd.read_csv(DATA_CSV)
    enc, is_bge = _encoder()
    tag = "bge" if is_bge else "minilm"
    cache = DATA_CSV.parent / f"job_embeddings_{tag}.npy"
    index_path = DATA_CSV.parent / f"job_faiss_{tag}.index"
    if cache.exists() and len(emb := np.load(cache)) == len(df):
        return df, emb, _faiss_index(emb, index_path)
    print(f"Encoding {len(df):,} job descriptions (one-off, cached) ...")
    emb = enc.encode(df["clean_description"].tolist(), batch_size=64,
                     convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=True)
    np.save(cache, emb)
    if index_path.exists():
        index_path.unlink()
    return df, emb, _faiss_index(emb, index_path)


def get_available_roles() -> list[str]:
    """All job categories a candidate can target."""
    df = pd.read_csv(DATA_CSV, usecols=["final_category"])
    return sorted(df["final_category"].dropna().unique())


def _skill_list(skills: str | list[str]) -> list[str]:
    parts = skills.split(",") if isinstance(skills, str) else skills
    return [s.strip() for s in parts if s.strip()]


def _candidate_text(skills, experience: str) -> str:
    return f"Skills: {', '.join(_skill_list(skills))}. Experience: {experience}"


def _encode_candidate(skills, experience: str) -> np.ndarray:
    enc, is_bge = _encoder()
    text = _candidate_text(skills, experience)
    text = BGE_QUERY_PREFIX + text if is_bge else text
    return enc.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


def _query_keywords(skills, experience: str) -> set[str]:
    text = _candidate_text(skills, experience).lower()
    tokens = re.findall(r"[a-z][a-z0-9+#.]{2,}", text)
    return {t.strip(".") for t in tokens if t.strip(".") not in _QUERY_STOPWORDS}


def _keyword_hits(keywords: set[str], descriptions: list[str]) -> int:
    if not keywords:
        return 0
    hits = 0
    for desc in descriptions:
        low = str(desc).lower()
        if any(k in low for k in keywords):
            hits += 1
    return hits


def _skill_split(skills: list[str], description: str) -> tuple[list[str], list[str]]:
    have = {s.lower() for s in skills}
    matched = [s for s in skills if skill_present(s, description)]
    missing = [j for j in extract_skills(description) if j.lower() not in have]
    return matched, missing


def _skill_overlap_score(skills: list[str], description: str) -> float:
    if not skills:
        return 0.0
    return sum(skill_present(s, description) for s in skills) / len(skills)


def _years_from_text(text: str, candidate: bool = False) -> float | None:
    """Extract explicit years of experience only.

    Clear entry-level phrases are treated as low experience. We intentionally
    do not infer experience from words like senior, manager, lead, or director
    because those words can appear in unrelated contexts and make the
    recommender over-filter good matches.
    """
    low = (text or "").lower()
    range_match = re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:years?|yrs?)\b", low)
    if range_match:
        return float(range_match.group(1))
    years = [float(n) for n in re.findall(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", low)]
    if years:
        return max(years) if candidate else min(years)
    if any(re.search(r"\brecent(?:ly)?\b.*\b(graduat(?:e|ed|ing))\b", line.lower())
           for line in (text or "").splitlines()):
        return 1.0 if candidate else 0.0
    entry_patterns = (
        r"\brecent\s+graduate\b",
        r"\brecently\s+graduat(?:ed|ing)\b",
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
        r"\b0\s*[-–]\s*1\s*(?:years?|yrs?)\b",
    )
    if any(re.search(pattern, low) for pattern in entry_patterns):
        return 1.0 if candidate else 0.0
    senior_patterns = (
        r"\bsenior[-\s]?level\b",
        r"\bsr\.?\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|consultant|specialist|architect|administrator|manager)\b",
        r"\bsenior\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|consultant|specialist|architect|administrator|manager|role|position)\b",
        r"\bexperienced\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|consultant|specialist|architect|administrator|professional)\b",
    )
    lead_patterns = (
        r"\blead\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|consultant|specialist|architect|administrator)\b",
        r"\bstaff\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|architect)\b",
        r"\bprincipal\s+(?:\w+\s+){0,3}(developer|engineer|analyst|scientist|consultant|architect)\b",
    )
    if any(re.search(pattern, low) for pattern in lead_patterns):
        return 6.0
    if any(re.search(pattern, low) for pattern in senior_patterns):
        return 4.0
    return None


def _experience_score(candidate_years: float | None, description: str) -> tuple[float, float | None]:
    """Experience fit score.

    Unknown years are neutral, not perfect, so experience has the same
    priority as skills without over-rewarding missing information.
    """
    required = _years_from_text(description, candidate=False)
    if candidate_years is None or required is None:
        return 0.5, required
    gap = required - candidate_years
    if gap <= 0:
        return 1.0, required
    if gap <= 1:
        return 0.85, required
    if gap <= 3:
        return 0.55, required
    return 0.25, required


def _role_pool(df: pd.DataFrame, target_roles) -> pd.Series:
    return (df["final_category"].isin(list(target_roles)) if target_roles
            else pd.Series(True, index=df.index))


def _ranked_hits(emb, query, mask_arr, index, min_score, max_hits: int | None = None):
    faiss = _try_faiss()
    if index is not None and faiss is not None:
        search_k = min(index.ntotal, max((max_hits or SEARCH_RETRIEVAL_POOL) * 4, GUARDRAIL_POOL_SIZE))
        while True:
            scores, ids = index.search(query.reshape(1, -1).astype("float32"), search_k)
            hits = [(int(g), float(s)) for g, s in zip(ids[0], scores[0])
                    if g >= 0 and mask_arr[g] and float(s) >= min_score]
            if max_hits is None or len(hits) >= max_hits or search_k >= index.ntotal:
                break
            search_k = min(index.ntotal, search_k * 2)
    else:
        sims = emb[mask_arr] @ query
        sub_idx = np.nonzero(mask_arr)[0]
        hits = [(int(sub_idx[p]), float(sims[p]))
                for p in range(len(sims)) if float(sims[p]) >= min_score]
    hits.sort(key=lambda gs: (-round(gs[1], 4), gs[0]))
    return hits[:max_hits] if max_hits is not None else hits


def _passes_relevance_guardrail(df, emb, query, mask_arr, index, skills, experience) -> bool:
    raw_hits = _ranked_hits(emb, query, mask_arr, index, -1.0, max_hits=GUARDRAIL_POOL_SIZE)
    if not raw_hits:
        return False
    if raw_hits[0][1] < MIN_GUARDRAIL_SIMILARITY:
        return False
    keywords = _query_keywords(skills, experience)
    descriptions = [df.iloc[gidx]["clean_description"] for gidx, _ in raw_hits]
    return _keyword_hits(keywords, descriptions) > 0


def match_embedding(skills, target_roles, experience, min_score: float = 0.0) -> list[dict]:
    """Fast Search: semantic embedding search."""
    df, emb, _ = _data()
    mask = _role_pool(df, target_roles)
    sub, skl = df[mask], _skill_list(skills)
    sims = emb[mask.to_numpy()] @ _encode_candidate(skills, experience)

    out = []
    for pos in np.argsort(-sims):
        score = float(sims[pos])
        if score < min_score:
            break
        row = sub.iloc[pos]
        out.append({
            "job_id": row["job_id"],
            "category": row["final_category"],
            "score": round(float(score), 4),
            "semantic_score": round(score, 4),
            "experience_score": 1.0,
            "matched_skills": [],
            "missing_skills": [],
            "skill_score": 0.0,
            "reason": f"semantic match {score:.2f}; ?/{len(skl)} skills present",
        })
        if len(out) >= 50:
            break

    for job in out:
        row_idx = sub[sub["job_id"] == job["job_id"]].index[0]
        desc = sub.loc[row_idx, "clean_description"]
        m, miss = _skill_split(skl, desc)
        if not m:
            continue
        job["matched_skills"] = m
        job["missing_skills"] = miss
        job["skill_score"] = round(_skill_overlap_score(skl, desc), 4)
        job["reason"] = (f"semantic match {job['semantic_score']:.2f}; "
                         f"{len(m)}/{len(skl)} skills present"
                         + (f" ({', '.join(m[:5])})" if m else ""))
    filtered = [job for job in out if job["matched_skills"]]
    filtered.sort(key=lambda d: (-d["score"], -d["semantic_score"], d["job_id"]))
    return filtered


def _best_match_candidates(skills, target_roles, experience, min_score: float = 0.0) -> list[dict]:
    """Best Match candidate pool: relevance guardrail + matched skills."""
    df, emb, index = _data()
    mask = _role_pool(df, target_roles)
    sub, skl = df[mask], _skill_list(skills)
    if not skl:
        return []
    query = _encode_candidate(skills, experience)
    candidate_years = _years_from_text(experience, candidate=True)
    effective_min_score = max(min_score, MIN_RECOMMENDATION_SCORE)
    if not _passes_relevance_guardrail(df, emb, query, mask.to_numpy(), index, skills, experience):
        return []

    out = []
    for gidx, score in _ranked_hits(
            emb, query, mask.to_numpy(), index, effective_min_score,
            max_hits=SEARCH_RETRIEVAL_POOL):
        row = df.iloc[gidx]
        exp_score, required_years = _experience_score(candidate_years, row["clean_description"])
        out.append({
            "job_id": row["job_id"],
            "category": row["final_category"],
            "score": round(float(score), 4),
            "semantic_score": round(score, 4),
            "experience_score": round(exp_score, 4),
            "candidate_years": candidate_years,
            "required_years": required_years,
            "matched_skills": [],
            "missing_skills": [],
            "reason": f"semantic match {score:.2f}; ?/{len(skl)} skills present",
        })
    out.sort(key=lambda d: (-d["score"], d["job_id"]))

    filtered = []
    for job in out:
        row_idx = sub[sub["job_id"] == job["job_id"]].index[0]
        desc = sub.loc[row_idx, "clean_description"]
        m, miss = _skill_split(skl, desc)
        if not m:
            continue
        skill_score = _skill_overlap_score(skl, desc)
        job["matched_skills"] = m
        job["missing_skills"] = miss
        job["skill_score"] = round(skill_score, 4)
        exp_note = ""
        if job.get("candidate_years") is not None and job.get("required_years") is not None:
            exp_note = f"; experience {job['candidate_years']:.0f}/{job['required_years']:.0f} years"
        job["reason"] = (f"semantic match {job['semantic_score']:.2f}; "
                         f"semantic score {job['score']:.2f}; "
                         f"{len(m)}/{len(skl)} skills present"
                         f"{exp_note}"
                         + (f" ({', '.join(m[:5])})" if m else ""))
        filtered.append(job)
    filtered.sort(key=lambda d: (
        -((d.get("skill_score", 0.0) + d.get("experience_score", 0.5)) / 2),
        -d.get("semantic_score", 0.0),
        d["job_id"],
    ))
    filtered = filtered[:50]
    return filtered


def match_hybrid(skills, target_roles, experience, top_k: int = 20) -> list[dict]:
    """Best Match: Fast Search candidates + skill coverage + experience fit."""
    base = match_embedding(skills, target_roles, experience)[:HYBRID_RETRIEVAL_POOL]
    if not base:
        return []
    df, _, _ = _data()
    desc = df.set_index("job_id")["clean_description"]
    skl = _skill_list(skills)
    candidate_years = _years_from_text(experience, candidate=True)
    ranked = []
    for item in base:
        job_desc = desc[item["job_id"]]
        skill_score = _skill_overlap_score(skl, job_desc)
        if skill_score <= 0:
            continue
        exp_score, required_years = _experience_score(candidate_years, job_desc)
        item["embedding_score"] = item.get("semantic_score", item["score"])
        item["skill_score"] = round(skill_score, 4)
        item["experience_score"] = round(exp_score, 4)
        item["candidate_years"] = candidate_years
        item["required_years"] = required_years
        equal_fit = (skill_score + item.get("experience_score", 0.5)) / 2
        item["score"] = round((0.80 * equal_fit) + (0.20 * item["embedding_score"]), 4)
        exp_note = ""
        if item.get("candidate_years") is not None and item.get("required_years") is not None:
            exp_note = f"; experience {item['candidate_years']:.0f}/{item['required_years']:.0f} years"
        item["reason"] = (f"best match score {item['score']:.2f}; "
                          f"skill coverage {skill_score:.2f}; "
                          f"experience fit {item.get('experience_score', 0.5):.2f}; "
                          f"semantic match {item['embedding_score']:.2f}"
                          f"{exp_note}")
        ranked.append(item)
    ranked = [item for item in ranked if item["score"] >= MIN_RECOMMENDATION_SCORE]
    ranked.sort(key=lambda d: (
        -((d["skill_score"] + d.get("experience_score", 0.5)) / 2),
        -d["skill_score"],
        -d.get("experience_score", 0.5),
        -d["embedding_score"],
        d["job_id"],
    ))
    return ranked[:top_k]


def recommend_jobs(skills, target_roles, experience, strategy: str = "embedding",
                   top_k: int = 10) -> list[dict]:
    fn = {"hybrid": match_hybrid}.get(strategy)
    return (fn(skills, target_roles, experience, top_k=top_k) if fn
            else match_embedding(skills, target_roles, experience)[:top_k])


def _metrics(ranked_ids: list, relevant: set, k: int) -> tuple[float, float, float]:
    hits = len(set(ranked_ids[:k]) & relevant)
    mrr = next((1 / (i + 1) for i, j in enumerate(ranked_ids) if j in relevant), 0.0)
    return hits / k, (hits / len(relevant) if relevant else 0.0), mrr


def benchmark_recommend(test_cases: list[dict], k: int = 5) -> dict:
    strategies = {
        "embedding": lambda c: match_embedding(c["skills"], c["target_roles"], c["experience"]),
        "hybrid": lambda c: match_hybrid(c["skills"], c["target_roles"], c["experience"], top_k=20),
    }
    result = {}
    for name, fn in strategies.items():
        per_case = [_metrics([m["job_id"] for m in fn(c)], set(c["relevant_job_ids"]), k)
                    for c in test_cases]
        p, r, m = (float(np.mean([row[i] for row in per_case])) for i in range(3))
        result[name] = {"precision": round(p, 4), "recall": round(r, 4), "mrr": round(m, 4)}
    table = pd.DataFrame(result).T.rename_axis("strategy").rename(
        columns={"precision": f"Precision@{k}", "recall": f"Recall@{k}", "mrr": "MRR"})
    print(f"\nRecommendation benchmark ({len(test_cases)} test cases, k={k})")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "recommendation_benchmark.csv")
    print(f"Saved -> {REPORTS_DIR / 'recommendation_benchmark.csv'}")
    return result


_DEMO_PROFILES = [
    dict(skills="Python, SQL, Airflow, Spark, ETL", target_roles=["Data Engineer"], experience="4 years building data pipelines"),
    dict(skills="penetration testing, SIEM, incident response, firewall", target_roles=["Cybersecurity Analyst"], experience="3 years in a SOC"),
    dict(skills="Kubernetes, Docker, Terraform, CI/CD, AWS", target_roles=["DevOps Engineer", "Cloud Engineer"], experience="5 years automating cloud infrastructure"),
    dict(skills="machine learning, PyTorch, NLP, statistics", target_roles=["Data Scientist", "AI Engineer"], experience="2 years training ML models"),
]


def demo_test_cases() -> list[dict]:
    df, _, _ = _data()
    cases = []
    for p in _DEMO_PROFILES:
        pool = df[_role_pool(df, p["target_roles"])]
        skl = [s.lower() for s in _skill_list(p["skills"])]
        rel = [r["job_id"] for _, r in pool.iterrows()
               if sum(s in r["clean_description"].lower() for s in skl) >= 3]
        cases.append({**p, "relevant_job_ids": rel})
    return cases


if __name__ == "__main__":
    print(f"{len(get_available_roles())} target roles available")
    benchmark_recommend(demo_test_cases())
