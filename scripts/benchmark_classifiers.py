"""
Job Classification Benchmark Pipeline
=====================================

Benchmarks three text-classification approaches on Saudi job postings,
using ONLY the job description text as the feature (job titles are
deliberately excluded to avoid label leakage — most labels were derived
from titles):

    A) TF-IDF (10k feats, 1-2 grams)  + LogisticRegression(balanced)
    B) Sentence-Transformer all-MiniLM-L6-v2 embeddings + same LogReg
    C) Fine-tuned distilbert-base-uncased (HF Trainer, early stopping)

Pipeline
--------
1. Merge final_labels_resolved.csv with job_data_saudi_clean.csv on job_id.
2. Clean text (URLs, emails, phones, non-printables, whitespace).
3. Drop exact-duplicate descriptions (prevents train/test leakage).
4. Keep categories with >= 50 examples.
5. Stratified 70/15/15 train/val/test split (seed 42).
6. Train + evaluate all three models on the held-out test set.
7. Log everything to Weights & Biases (project "job-classification")
   when an API key is available; otherwise runs fully offline and
   prints the comparison table.
8. Save the best model (by test macro-F1) to models/best_model/ and
   the split indices to data/processed/data_splits.pkl.

Usage
-----
    python scripts/benchmark_classifiers.py

Outputs
-------
    data/processed/cleaned_data.csv     cleaned + filtered dataset
    data/processed/data_splits.pkl      train/val/test indices + metadata
    reports/confusion_matrix_<model>.{csv,png}
    models/best_model/                  best model + metadata
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_ROOT / "final_labels_resolved.csv"
JOBS_CSV = PROJECT_ROOT / "job_data_saudi_clean.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
MIN_EXAMPLES_PER_CLASS = 50          # keep categories with >= this many rows
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Model A — TF-IDF + Logistic Regression
TFIDF_PARAMS = dict(max_features=10_000, ngram_range=(1, 2), stop_words="english")
LOGREG_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=SEED)

# Model B — Sentence-Transformer embeddings + Logistic Regression
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ST_ENCODE_BATCH = 64

# Model C — fine-tuned transformer
HF_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 12
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 2

WANDB_PROJECT = "job-classification"


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Conservative phone pattern: 9+ digit-ish sequences with separators,
# optionally prefixed with +. Anchored on digits at both ends so it does
# not eat ordinary numbers like "5 years" or "2026".
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{7,}\d(?![\w.])")
# ASCII/Unicode control characters and zero-width marks
NONPRINTABLE_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\u00a0\u00ad\u200b-\u200f\u2028\u2029\ufeff]"
)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a raw job description for modelling.

    Removes URLs, email addresses, phone numbers, and non-printable
    characters, then collapses all whitespace runs to single spaces.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = NONPRINTABLE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Data loading / preparation
# --------------------------------------------------------------------------

def load_and_prepare_data() -> pd.DataFrame:
    """Merge labels with descriptions, clean text, dedupe, filter classes.

    Returns a DataFrame with columns [job_id, clean_description,
    final_category] and a fresh 0..n-1 index (the split indices saved
    later refer to this index / to data/processed/cleaned_data.csv rows).
    """
    print("\n[1/6] Loading and merging data ...")
    labels = pd.read_csv(LABELS_CSV)                       # job_id, job_title, final_category
    jobs = pd.read_csv(JOBS_CSV, usecols=["job_id", "job_description"])

    df = labels.merge(jobs, on="job_id", how="inner")
    df = df.dropna(subset=["final_category", "job_description"])
    print(f"    merged rows: {len(df):,}")

    print("[2/6] Cleaning descriptions ...")
    tqdm.pandas(desc="    cleaning")
    df["clean_description"] = df["job_description"].progress_apply(clean_text)

    # Guard against rows that became empty after cleaning
    df = df[df["clean_description"].str.len() > 0]

    # Exact-duplicate descriptions would leak across the train/test
    # boundary under a random split, inflating scores — keep first only.
    n_dup = df["clean_description"].duplicated().sum()
    df = df.drop_duplicates(subset="clean_description", keep="first")
    print(f"    dropped {n_dup} exact-duplicate descriptions")

    # Keep only categories with enough support to learn / evaluate
    counts = df["final_category"].value_counts()
    keep = counts[counts >= MIN_EXAMPLES_PER_CLASS].index
    df = df[df["final_category"].isin(keep)].reset_index(drop=True)
    print(
        f"    kept {len(keep)} categories with >= {MIN_EXAMPLES_PER_CLASS} "
        f"examples -> {len(df):,} rows"
    )

    out_cols = ["job_id", "clean_description", "final_category"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(PROCESSED_DIR / "cleaned_data.csv", index=False)
    print(f"    saved cleaned dataset -> {PROCESSED_DIR / 'cleaned_data.csv'}")
    return df[out_cols]


def create_splits(df: pd.DataFrame) -> dict:
    """Stratified 70/15/15 split. Returns dict of positional indices.

    The indices refer to row positions in `df` (== row order of
    data/processed/cleaned_data.csv). The full dict, including job_ids
    and the label vocabulary, is pickled for downstream reproducibility.
    """
    print("[3/6] Creating stratified 70/15/15 split ...")
    y = df["final_category"].values
    idx = np.arange(len(df))

    # First carve off train, then split the remainder 50/50 into val/test.
    train_idx, temp_idx = train_test_split(
        idx, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y[temp_idx],
        random_state=SEED,
    )

    splits = {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
        "train_job_ids": df.loc[np.sort(train_idx), "job_id"].tolist(),
        "val_job_ids": df.loc[np.sort(val_idx), "job_id"].tolist(),
        "test_job_ids": df.loc[np.sort(test_idx), "job_id"].tolist(),
        "classes": sorted(df["final_category"].unique().tolist()),
        "seed": SEED,
    }
    with open(PROCESSED_DIR / "data_splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    print(
        f"    train={len(train_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}  -> saved data_splits.pkl"
    )
    return splits


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, class_names, model_key) -> dict:
    """Compute accuracy / macro-F1, save confusion matrix as CSV + PNG."""
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(REPORTS_DIR / f"confusion_matrix_{model_key}.csv")

    # Heatmap (row-normalised so small classes remain visible)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        cm_norm, xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", vmin=0, vmax=1, square=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title(f"{model_key} — confusion matrix (row-normalised)\n"
                 f"accuracy={acc:.4f}  macro-F1={f1m:.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrix_{model_key}.png", dpi=150)
    plt.close(fig)

    return {"accuracy": acc, "f1_macro": f1m, "confusion_matrix": cm}


def wandb_is_available() -> bool:
    """True when a W&B API key is configured (env var or netrc login)."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc
        netrc_path = Path.home() / ("_netrc" if os.name == "nt" else ".netrc")
        return netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai") is not None
    except Exception:
        return False


def log_to_wandb(use_wandb, run_name, config, results, y_true, y_pred, class_names):
    """Log one W&B run per model with hyperparams, metrics and CM.

    Reuses the active run when one exists (the HF Trainer opens its own
    run for model C) so training curves and test metrics land together.
    """
    if not use_wandb:
        return
    import wandb

    run = wandb.run
    if run is None:
        run = wandb.init(project=WANDB_PROJECT, name=run_name, config=config)
    elif config:
        run.config.update(config, allow_val_change=True)
    run.log({
        "test/accuracy": results["accuracy"],
        "test/f1_macro": results["f1_macro"],
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=np.asarray(y_true), preds=np.asarray(y_pred),
            class_names=list(class_names),
        ),
    })
    cm_png = REPORTS_DIR / f"confusion_matrix_{run_name}.png"
    if cm_png.exists():
        run.log({"test/confusion_matrix_heatmap": wandb.Image(str(cm_png))})
    run.finish()


# --------------------------------------------------------------------------
# Model A — TF-IDF + Logistic Regression
# --------------------------------------------------------------------------

def run_tfidf_logreg(texts, y, splits, class_names, use_wandb) -> dict:
    print("\n[4/6] Model A: TF-IDF + LogisticRegression")
    tr, te = splits["train_idx"], splits["test_idx"]
    t0 = time.time()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", LogisticRegression(**LOGREG_PARAMS)),
    ])
    # Validation set is unused here (no hyperparameter search for the
    # baselines) — it exists so all three models share identical test data.
    pipe.fit(texts[tr], y[tr])
    y_pred = pipe.predict(texts[te])

    results = evaluate_predictions(y[te], y_pred, class_names, "tfidf_logreg")
    results["train_seconds"] = time.time() - t0
    results["artifact"] = pipe
    print(f"    accuracy={results['accuracy']:.4f}  "
          f"macro-F1={results['f1_macro']:.4f}  ({results['train_seconds']:.0f}s)")

    log_to_wandb(
        use_wandb, "tfidf_logreg",
        {"model": "tfidf+logreg", **TFIDF_PARAMS, **LOGREG_PARAMS,
         "ngram_range": str(TFIDF_PARAMS["ngram_range"])},
        results, y[te], y_pred, class_names,
    )
    return results


# --------------------------------------------------------------------------
# Model B — Sentence-Transformer embeddings + Logistic Regression
# --------------------------------------------------------------------------

def run_st_logreg(texts, y, splits, class_names, use_wandb) -> dict:
    print("\n[5/6] Model B: all-MiniLM-L6-v2 embeddings + LogisticRegression")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    encoding {len(texts):,} descriptions on {device} ...")
    t0 = time.time()

    encoder = SentenceTransformer(ST_MODEL_NAME, device=device)
    # Encode the full corpus once; rows are selected per split afterwards.
    # MiniLM truncates internally at 256 word pieces.
    emb = encoder.encode(
        list(texts), batch_size=ST_ENCODE_BATCH,
        show_progress_bar=True, convert_to_numpy=True,
    )

    tr, te = splits["train_idx"], splits["test_idx"]
    clf = LogisticRegression(**LOGREG_PARAMS)
    clf.fit(emb[tr], y[tr])
    y_pred = clf.predict(emb[te])

    results = evaluate_predictions(y[te], y_pred, class_names, "minilm_logreg")
    results["train_seconds"] = time.time() - t0
    results["artifact"] = {"encoder_name": ST_MODEL_NAME, "classifier": clf}
    print(f"    accuracy={results['accuracy']:.4f}  "
          f"macro-F1={results['f1_macro']:.4f}  ({results['train_seconds']:.0f}s)")

    log_to_wandb(
        use_wandb, "minilm_logreg",
        {"model": "minilm+logreg", "encoder": ST_MODEL_NAME, **LOGREG_PARAMS},
        results, y[te], y_pred, class_names,
    )
    return results


# --------------------------------------------------------------------------
# Model C — fine-tuned DistilBERT
# --------------------------------------------------------------------------

def run_finetuned_transformer(texts, y, splits, class_names, use_wandb) -> dict:
    print("\n[6/6] Model C: fine-tuned distilbert-base-uncased")
    import inspect

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    fine-tuning on {device}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: c for i, c in enumerate(class_names)},
        label2id={c: i for i, c in enumerate(class_names)},
    )

    class JobDataset(torch.utils.data.Dataset):
        """Pre-tokenised dataset; dynamic padding via DataCollator."""

        def __init__(self, split_texts, split_labels):
            self.encodings = tokenizer(
                list(split_texts), truncation=True, max_length=MAX_LENGTH
            )
            self.labels = list(split_labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    tr, va, te = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    train_ds = JobDataset(texts[tr], y[tr])
    val_ds = JobDataset(texts[va], y[va])
    test_ds = JobDataset(texts[te], y[te])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # `evaluation_strategy` was renamed to `eval_strategy` in newer
    # transformers releases — pick whichever this install supports.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "distilbert_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LEARNING_RATE,
        **{strategy_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,           # restore best-val checkpoint
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        bf16=(device == "cuda"),               # fast mixed precision on RTX GPUs
        logging_steps=50,
        seed=SEED,
        report_to=(["wandb"] if use_wandb else []),
        run_name="distilbert_finetune",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)

    results = evaluate_predictions(y[te], y_pred, class_names, "distilbert_finetune")
    results["train_seconds"] = time.time() - t0
    results["artifact"] = {"trainer": trainer, "tokenizer": tokenizer}
    print(f"    accuracy={results['accuracy']:.4f}  "
          f"macro-F1={results['f1_macro']:.4f}  ({results['train_seconds']:.0f}s)")

    log_to_wandb(   # joins the Trainer's open run, adds test metrics + CM
        use_wandb, "distilbert_finetune",
        {"model": HF_MODEL_NAME, "max_length": MAX_LENGTH,
         "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
         "learning_rate": LEARNING_RATE,
         "early_stopping_patience": EARLY_STOPPING_PATIENCE},
        results, y[te], y_pred, class_names,
    )

    # Print the per-class report for the (usually) strongest model
    print("\n    Per-class test report (DistilBERT):")
    print(classification_report(
        y[te], y_pred, target_names=class_names, zero_division=0, digits=3,
    ))
    return results


# --------------------------------------------------------------------------
# Best-model persistence
# --------------------------------------------------------------------------

def save_best_model(best_key, results, class_names):
    """Persist the winning model + metadata to models/best_model/."""
    import joblib

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": best_key,
        "test_accuracy": results["accuracy"],
        "test_f1_macro": results["f1_macro"],
        "classes": list(class_names),
        "feature": "job_description (cleaned)",
        "seed": SEED,
    }

    if best_key == "tfidf_logreg":
        joblib.dump(results["artifact"], BEST_MODEL_DIR / "tfidf_logreg.joblib")
        meta["usage"] = "joblib.load(...).predict([clean_text(desc)])"
    elif best_key == "minilm_logreg":
        art = results["artifact"]
        joblib.dump(art["classifier"], BEST_MODEL_DIR / "minilm_logreg.joblib")
        meta["encoder"] = art["encoder_name"]
        meta["usage"] = (
            "emb = SentenceTransformer(meta['encoder']).encode([clean_text(desc)]); "
            "joblib.load(...).predict(emb)"
        )
    else:  # distilbert_finetune
        art = results["artifact"]
        art["trainer"].save_model(str(BEST_MODEL_DIR))
        art["tokenizer"].save_pretrained(str(BEST_MODEL_DIR))
        meta["usage"] = (
            "pipeline('text-classification', model='models/best_model', "
            "truncation=True, max_length=256)"
        )

    with open(BEST_MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nBest model ({best_key}) saved -> {BEST_MODEL_DIR}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    use_wandb = wandb_is_available()
    if use_wandb:
        os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        print(f"W&B logging ENABLED -> project '{WANDB_PROJECT}'")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B API key not found -> running without W&B "
              "(metrics printed to console only)")

    # ---- data ------------------------------------------------------------
    df = load_and_prepare_data()
    splits = create_splits(df)

    le = LabelEncoder().fit(splits["classes"])
    class_names = le.classes_
    texts = df["clean_description"].values
    y = le.transform(df["final_category"].values)

    # ---- models ----------------------------------------------------------
    all_results = {}
    all_results["tfidf_logreg"] = run_tfidf_logreg(texts, y, splits, class_names, use_wandb)
    all_results["minilm_logreg"] = run_st_logreg(texts, y, splits, class_names, use_wandb)
    all_results["distilbert_finetune"] = run_finetuned_transformer(
        texts, y, splits, class_names, use_wandb
    )

    # ---- comparison table --------------------------------------------------
    table = pd.DataFrame(
        {
            "Model": [
                "A) TF-IDF + LogReg",
                "B) MiniLM emb + LogReg",
                "C) DistilBERT fine-tuned",
            ],
            "Accuracy": [all_results[k]["accuracy"]
                         for k in ("tfidf_logreg", "minilm_logreg", "distilbert_finetune")],
            "Macro F1": [all_results[k]["f1_macro"]
                         for k in ("tfidf_logreg", "minilm_logreg", "distilbert_finetune")],
            "Train time (s)": [round(all_results[k]["train_seconds"])
                               for k in ("tfidf_logreg", "minilm_logreg", "distilbert_finetune")],
        }
    )
    print("\n" + "=" * 64)
    print("FINAL COMPARISON (test set, n="
          f"{len(splits['test_idx']):,}, {len(class_names)} classes)")
    print("=" * 64)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 64)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # ---- save best ---------------------------------------------------------
    best_key = max(all_results, key=lambda k: all_results[k]["f1_macro"])
    save_best_model(best_key, all_results[best_key], class_names)


if __name__ == "__main__":
    main()
