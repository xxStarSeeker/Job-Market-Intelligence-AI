"""
FastAPI wrapper for job classification and recommendation. All model logic
lives in scripts/predict.py (tech-11 ensemble) and scripts/recommend.py.

    .venv\\Scripts\\uvicorn.exe app:app --reload      # docs at /docs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.predict import predict_batch, predict_category
from scripts.recommend import get_available_roles, match_embedding, recommend_jobs
from scripts.benchmark_classifiers import clean_text

app = FastAPI(title="Job Market Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=1)


class ClassifyBatchRequest(BaseModel):
    descriptions: list[str] = Field(min_length=1)


class RecommendRequest(BaseModel):
    skills: str | list[str]
    target_roles: list[str]
    experience: str = ""
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: Literal["embedding", "hybrid"] = "embedding"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "ensemble"}


@app.get("/roles")
def roles() -> dict:
    return {"roles": get_available_roles()}


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    """Classify one job description into one of the 11 tech categories."""
    return predict_category(clean_text(req.description))


@app.post("/classify/batch")
def classify_batch(req: ClassifyBatchRequest) -> dict:
    """Classify many descriptions at once (batch-encoded)."""
    return {"results": predict_batch(req.descriptions)}


@app.post("/recommend")
def recommend(req: RecommendRequest) -> list[dict]:
    """Match a candidate to jobs with Fast Search or Best Match."""
    if req.strategy == "embedding":
        return match_embedding(req.skills, req.target_roles, req.experience,
                               min_score=req.min_score)
    matches = recommend_jobs(req.skills, req.target_roles, req.experience,
                             strategy=req.strategy)
    return [m for m in matches if m["score"] >= req.min_score]


if __name__ == "__main__": import uvicorn; uvicorn.run("app:app", reload=True)  # noqa: E702
