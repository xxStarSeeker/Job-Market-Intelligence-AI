"""
Inference API for the tech-11 job-category ensemble in models/best_model/.

Classifies raw job descriptions into 11 tech categories with the
soft-voting ensemble benchmarked at 0.94 test accuracy. Inputs are cleaned
with the training-time clean_text; weights come from ensemble.json, class
order from metadata.json. Degrades gracefully: no GPU -> transformers on
CPU (warning); a component that fails to load is dropped and the rest
renormalised - worst case TF-IDF only.

    python scripts/predict.py --text "We need a Kubernetes engineer ..."
    python scripts/predict.py --csv new_jobs.csv [--output out.csv]
    from scripts.predict import predict_category, predict_batch
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.benchmark_classifiers import clean_text  # noqa: E402

BEST_MODEL_DIR = PROJECT_ROOT / "models" / "best_model"
BATCH_SIZE = 16

PredictFn = Callable[[list[str]], np.ndarray]   # cleaned texts -> (n, n_classes)


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


@functools.lru_cache(maxsize=None)
def _torch_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    _warn("GPU not available - transformer components will run on CPU (slower)")
    return "cpu"


def _load_tfidf(comp_dir: Path, info: dict) -> PredictFn:
    """TF-IDF + LogisticRegression pipeline; consumes cleaned text directly."""
    import joblib
    pipe = joblib.load(comp_dir / info["file"])
    return pipe.predict_proba


def _load_bge(comp_dir: Path, info: dict) -> PredictFn:
    """Chunked sentence-embedding classifier (length-weighted mean pooling)."""
    import joblib
    from sentence_transformers import SentenceTransformer

    clf = joblib.load(comp_dir / info["file"])
    encoder = SentenceTransformer(info["encoder"], device=_torch_device())
    tok, chunk = encoder.tokenizer, info["chunk_tokens"]

    def predict(texts: list[str]) -> np.ndarray:
        pieces, owners, lengths = [], [], []
        for di, t in enumerate(texts):
            ids = tok(t, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
            for s in range(0, len(ids), chunk):
                part = ids[s:s + chunk]
                pieces.append(tok.decode(part)); owners.append(di); lengths.append(len(part))
        emb = encoder.encode(pieces, batch_size=64, convert_to_numpy=True,
                             normalize_embeddings=True)
        X = np.zeros((len(texts), emb.shape[1]))
        wsum = np.zeros(len(texts))
        for e, o, w in zip(emb, owners, lengths):
            X[o] += e * w
            wsum[o] += w
        X /= wsum[:, None]
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return clf.predict_proba(X)

    return predict


def _load_hf(comp_dir: Path, info: dict) -> PredictFn:
    """Fine-tuned HuggingFace sequence classifier saved under comp_dir."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _torch_device()
    tok = AutoTokenizer.from_pretrained(comp_dir)
    model = AutoModelForSequenceClassification.from_pretrained(comp_dir).to(device).eval()
    max_len = info["max_length"]

    def predict(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                enc = tok(texts[i:i + BATCH_SIZE], truncation=True,
                          max_length=max_len, padding=True,
                          return_tensors="pt").to(device)
                logits = model(**enc).logits.float()
                out.append(torch.softmax(logits, -1).cpu().numpy())
        return np.vstack(out)

    return predict


def _loader_for(info: dict) -> Callable[[Path, dict], PredictFn]:
    return (_load_hf if "base_model" in info
            else _load_bge if "encoder" in info else _load_tfidf)


class _Ensemble:
    """Weighted soft vote over whichever components load successfully."""

    def __init__(self, model_dir: Path = BEST_MODEL_DIR):
        meta = json.loads((model_dir / "metadata.json").read_text("utf-8"))
        ens = json.loads((model_dir / "ensemble.json").read_text("utf-8"))
        self.classes: list[str] = meta["classes"]
        self.parts: list[tuple[float, PredictFn]] = []
        for name, info in ens["components"].items():
            try:
                self.parts.append(
                    (info["weight"], _loader_for(info)(model_dir / name, info)))
            except Exception as exc:  # degrade per component, don't die
                _warn(f"component '{name}' failed to load ({exc!r}) - skipping it")
        if not self.parts:
            raise RuntimeError(f"no ensemble component could be loaded from {model_dir}")
        if len(self.parts) < len(ens["components"]):
            _warn(f"running with {len(self.parts)}/{len(ens['components'])} components"
                  " - expect accuracy below the benchmarked 0.94")
        total = sum(w for w, _ in self.parts)
        self.parts = [(w / total, fn) for w, fn in self.parts]

    def predict_probs(self, raw_texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in raw_texts]
        return sum(w * fn(cleaned) for w, fn in self.parts)


@functools.lru_cache(maxsize=None)
def _predictor() -> _Ensemble:
    return _Ensemble()


def predict_batch(texts: list[str]) -> list[dict]:
    """Classify raw job descriptions in bulk; one result dict per input."""
    if not texts:
        return []
    pred = _predictor()
    out = []
    for row in pred.predict_probs(list(texts)):
        i = int(row.argmax())
        out.append({"predicted_category": pred.classes[i],
                    "confidence": float(row[i]),
                    "all_scores": {c: float(p) for c, p in zip(pred.classes, row)}})
    return out


def predict_category(text: str) -> dict:
    """Classify one raw job description (see module docstring for shape)."""
    return predict_batch([text])[0]


def _run_csv(path: Path, out_path: Path | None) -> None:
    import pandas as pd
    df = pd.read_csv(path)
    if "job_description" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'job_description' column")
    results = predict_batch(df["job_description"].fillna("").tolist())
    df["predicted_category"] = [r["predicted_category"] for r in results]
    df["confidence"] = [round(r["confidence"], 4) for r in results]
    out_path = out_path or path.with_name(path.stem + "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} predictions -> {out_path}")


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Classify job descriptions with the tech-11 ensemble.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="a single job-description string")
    src.add_argument("--csv", type=Path, help="CSV with a 'job_description' column")
    ap.add_argument("--output", type=Path,
                    help="output path for --csv (default: <input>_predictions.csv)")
    args = ap.parse_args()
    if args.text is not None:
        res = predict_category(args.text)
        top3 = sorted(res["all_scores"].items(), key=lambda kv: -kv[1])[:3]
        print(f"predicted_category: {res['predicted_category']}")
        print(f"confidence:         {res['confidence']:.4f}")
        print("top 3:              " + ", ".join(f"{c} ({p:.3f})" for c, p in top3))
    else:
        _run_csv(args.csv, args.output)


if __name__ == "__main__":
    _cli()
