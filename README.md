# TalentFit — Job Market Intelligence System

**SDA AI Engineering Bootcamp · Group 4** (with WeCloudData)
Abdulwahab Almusharraf · Raghad Khan · Thamer Al Otaibi · Yazeed Alshawmer

TalentFit is a two-part AI system for the job market:

1. **Classification** — reads a job posting and predicts its role category (11 tech roles)
   with a soft-voting ensemble (TF-IDF + BGE embeddings + DistilBERT + ModernBERT),
   ~0.94 test accuracy.
2. **Recommendation** — takes a resume, finds the best-fit jobs by semantic + skill-aware
   matching, and explains each match (matched skills, missing skills, a plain-language reason).
   Two strategies: **Fast Search** (embedding) and **Best Match** (hybrid). Both run fully
   offline — no LLM needed at recommendation time.

Both models are served by a small **FastAPI** API, with a **Streamlit** demo where you can
upload a resume and see ranked, explained matches. Job categories were produced once, offline,
by an **LLM-labeling** pipeline (Ollama `qwen2.5:32b`, two passes + an LLM judge for
disagreements); that step is not needed to run the app.

---

## Folder layout

```
02_code/
├── 01_data/                     Source datasets
│   ├── raw/saudi_job_market.csv         ~47k raw Saudi postings (49 cols)
│   ├── job_data_saudi_clean.csv         cleaned postings (~13.9k)
│   └── labels/                          LLM labels + disagreement resolution
├── 02_src/                      Code (runnable as-is)
│   ├── app.py                           FastAPI service
│   ├── streamlit_app.py                 resume-upload demo UI
│   ├── scripts/                         predict.py, recommend.py, benchmarks, skills…
│   ├── Backend/                         LLM-labeling pipeline + config
│   ├── models/best_model/               trained ensemble (TF-IDF, BGE, DistilBERT, ModernBERT)
│   ├── data/processed/                  cleaned_data.csv, BGE embeddings, FAISS index, splits
│   ├── job_data_saudi_clean.csv         job metadata for result cards (title/company/location)
│   ├── notebooks/                       TalentFit_clean_and_eda.ipynb, manual_compare.ipynb
│   ├── cvs_test/                        validation on real resumes (88% correct)
│   ├── Dockerfile, .dockerignore        optional containerized run
│   └── requirements.txt                 (copy, for the Docker build context)
├── 03_assets/                   Confusion matrices, comparison tables, logo, app screenshots
├── requirements.txt
└── README.md                    (this file)
```

> Note: `01_data/` holds the **source** datasets. The **derived** artifacts the code loads at
> run time (cleaned_data.csv, BGE embeddings, FAISS index, train/val/test splits) ship inside
> `02_src/data/processed/` so the system runs without rebuilding anything.

---

## Requirements

- **Python 3.13** (a virtual environment is recommended)
- Packages in `requirements.txt`: numpy, pandas, scikit-learn, scipy, sentence-transformers,
  transformers, faiss-cpu, fastapi, uvicorn, pydantic, streamlit, PyPDF2, joblib, matplotlib,
  seaborn, tqdm, requests, aiohttp.
- **PyTorch** — installed separately so you get the right CPU/GPU build. A GPU is optional;
  the transformer models fall back to CPU automatically (slower).
- **Ollama** is **not** required to run the app. It is only used to *re-generate* the labeled
  dataset (`Backend/scripts/`), which is already provided.

## Install

```bash
# from inside 02_code/02_src/
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS/Linux
.venv/bin/pip install -r requirements.txt

# PyTorch — pick the build for your machine from https://pytorch.org
# CPU example:
.venv\Scripts\pip install torch
```

## Run (local — recommended)

All commands below are run from `02_code/02_src/` (paths are relative to that folder).

```bash
# 1) Inference API  → Swagger UI at http://127.0.0.1:8000/docs
.venv\Scripts\uvicorn app:app --reload

# 2) Classify one job description
.venv\Scripts\python scripts\predict.py --text "We need a Kubernetes engineer for CI/CD on AWS"

# 3) Recommendation benchmark (embedding vs hybrid)
.venv\Scripts\python scripts\recommend.py

# 4) Candidate demo UI (start the API first, then:)
.venv\Scripts\streamlit run streamlit_app.py
```

### API quick reference

| Method & path          | Purpose                                            |
|------------------------|----------------------------------------------------|
| `GET  /health`         | Liveness + active model                            |
| `GET  /roles`          | List the 37 targetable roles                       |
| `POST /classify`       | Classify one job description                       |
| `POST /classify/batch` | Classify many descriptions at once                 |
| `POST /recommend`      | Rank jobs for a candidate (`embedding` or `hybrid`)|

```bash
# Example
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{
  \"skills\": \"Python, SQL, Airflow, Spark, ETL\",
  \"target_roles\": [\"Data Engineer\"],
  \"experience\": \"4 years building data pipelines\",
  \"strategy\": \"hybrid\", \"min_score\": 0.3 }"
```

## Run with Docker (self-contained — two commands)

The `Dockerfile` in `02_src/` builds a **self-contained** image: the trained models, the
precomputed embeddings/FAISS index, the CPU build of PyTorch, and the BGE embedding model are all
baked in, so no volume mounts and no extra setup are needed.

```bash
# from 02_code/02_src/
docker build -t talentfit .
docker run -p 8000:8000 -p 8501:8501 talentfit

# optional: enable the "resume tips" LLM feature using Ollama running on the host
docker run -p 8000:8000 -p 8501:8501 -e OLLAMA_URL=http://host.docker.internal:11434 talentfit
```

Then open:
- **API + Swagger docs** → http://localhost:8000/docs
- **Streamlit demo** → http://localhost:8501

Notes: the first build downloads PyTorch + the embedding model and copies the trained models, so it
takes a few minutes and the image is large (~5 GB) — this is the trade-off for a one-command run
with nothing else to install. Build needs internet; once built, the container runs offline.
(Verified end-to-end: `docker build` + `docker run` → `/health`, `/classify`, `/recommend`, and the
Streamlit demo all work.)

## Configuration / environment

- **No paid API keys** are required. The core app (classify + resume→jobs recommendations) runs
  with **no LLM**.
- The only LLM-backed feature is the optional **"resume tips"** button. It calls a local **Ollama**
  model and **auto-detects whatever is installed** — `qwen2.5` 3b / 7b / 32b all work, no config
  needed (it prefers the largest installed qwen2.5, falling back to any available model).
- To use it: have Ollama running. Locally it's found at `http://localhost:11434` automatically;
  from **Docker**, pass `-e OLLAMA_URL=http://host.docker.internal:11434` (see above). Override the
  model with `OLLAMA_MODEL` if you want a specific tag.
- On first local run the BGE encoder (`BAAI/bge-base-en-v1.5`) downloads from Hugging Face (one-off,
  internet required); the Docker image already bakes it in. Trained classifier weights ship under
  `02_src/models/best_model/`.

## Results (held-out test set)

- **Classification:** ensemble **0.94 accuracy / 0.943 macro-F1** across 11 tech roles.
- **Recommendation:** **Best Match (hybrid)** — Precision@5 **0.90**, MRR **1.00**;
  Fast Search (embedding) — Precision@5 0.80, MRR 0.71.
- **Real-resume check:** **88%** correct on 60 real CVs from a public Hugging Face dataset.

See `03_assets/` for confusion matrices and the benchmark tables, and `03_project_report/`
for the full report.

## Limitations / known issues

- The 0.94 classifier covers the **11 tech roles**; non-tech postings are forced into the
  nearest tech class (the recommender spans all 37 roles).
- Recommendation metrics use a **synthetic** (skill-overlap) relevance set, not human labels.
- English only; very long resumes are truncated by the transformer token limits (512/1024).
