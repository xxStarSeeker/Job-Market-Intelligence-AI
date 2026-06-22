"""
Retrain ONLY the DistilBERT model (model C) with more epochs.
==============================================================

The original benchmark run trained DistilBERT for 3 epochs, which left it
undertrained (val macro-F1 was still climbing at the last epoch). This
script re-runs just the fine-tuning stage with the updated config in
benchmark_classifiers.py (NUM_EPOCHS / EARLY_STOPPING_PATIENCE), reusing:

    data/processed/cleaned_data.csv     (no re-cleaning)
    data/processed/data_splits.pkl      (no re-splitting — file untouched)

It then updates the DistilBERT row in reports/model_comparison.csv and,
only if DistilBERT now has the best test macro-F1, refreshes
models/best_model/.

Usage
-----
    .venv\\Scripts\\python.exe scripts/retrain_distilbert.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_classifiers as bc


def main():
    np.random.seed(bc.SEED)

    use_wandb = bc.wandb_is_available()
    if use_wandb:
        os.environ["WANDB_PROJECT"] = bc.WANDB_PROJECT
        print(f"W&B logging ENABLED -> project '{bc.WANDB_PROJECT}'")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B API key not found -> running without W&B")

    # ---- load the EXISTING cleaned data and splits -------------------------
    df = pd.read_csv(bc.PROCESSED_DIR / "cleaned_data.csv")
    with open(bc.PROCESSED_DIR / "data_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    # The pickled job_ids must line up with the csv rows the indices point
    # at — guards against the csv having been regenerated since the split.
    for name in ("train", "val", "test"):
        if df.loc[splits[f"{name}_idx"], "job_id"].tolist() != splits[f"{name}_job_ids"]:
            sys.exit(f"ERROR: {name} split indices no longer match "
                     "cleaned_data.csv — rerun the full benchmark instead.")
    print(f"Reusing saved splits: train={len(splits['train_idx']):,}  "
          f"val={len(splits['val_idx']):,}  test={len(splits['test_idx']):,}  "
          f"({len(splits['classes'])} classes)")

    le = LabelEncoder().fit(splits["classes"])
    class_names = le.classes_
    texts = df["clean_description"].values
    y = le.transform(df["final_category"].values)

    # ---- fine-tune (config comes from benchmark_classifiers constants) -----
    print(f"Config: epochs={bc.NUM_EPOCHS}  lr={bc.LEARNING_RATE}  "
          f"early-stopping patience={bc.EARLY_STOPPING_PATIENCE}")
    results = bc.run_finetuned_transformer(texts, y, splits, class_names, use_wandb)

    # ---- update the DistilBERT row of the comparison table -----------------
    cmp_path = bc.REPORTS_DIR / "model_comparison.csv"
    table = pd.read_csv(cmp_path)
    row = table["Model"].str.startswith("C)")
    table.loc[row, "Accuracy"] = results["accuracy"]
    table.loc[row, "Macro F1"] = results["f1_macro"]
    table.loc[row, "Train time (s)"] = round(results["train_seconds"])
    table.to_csv(cmp_path, index=False)

    print("\n" + "=" * 64)
    print(f"UPDATED COMPARISON (test set, n={len(splits['test_idx']):,}, "
          f"{len(class_names)} classes)")
    print("=" * 64)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 64)

    # ---- promote DistilBERT only if it now wins on test macro-F1 -----------
    best_row = table.loc[table["Macro F1"].idxmax()]
    if best_row["Model"].startswith("C)"):
        # clear stale baseline artifacts before saving the transformer
        for stale in ("tfidf_logreg.joblib", "minilm_logreg.joblib"):
            p = bc.BEST_MODEL_DIR / stale
            if p.exists():
                p.unlink()
                print(f"Removed stale {p}")
        bc.save_best_model("distilbert_finetune", results, class_names)
    else:
        print(f"\nDistilBERT (macro-F1 {results['f1_macro']:.4f}) did NOT beat "
              f"the best recorded model ({best_row['Model']}, "
              f"macro-F1 {best_row['Macro F1']:.4f}) — models/best_model/ "
              "left unchanged.")


if __name__ == "__main__":
    main()
