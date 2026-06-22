# Project Structure

Use the root project folder:

```text
C:\Users\ragoo\Desktop\SDA\projects\Job Market Intelligence AI
```

Do not run files from the duplicated nested folder:

```text
Job Market Intelligence AI\Job Market Intelligence AI
```

## Main Files

- `app.py` - FastAPI backend.
- `streamlit_app.py` - Streamlit interface.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Optional container setup.

## Source Code

- `scripts/recommend.py` - job recommendation logic.
- `scripts/extract_skills.py` - skill extraction and skill matching.
- `scripts/resume_evidence.py` - local Ollama resume evidence extraction.
- `scripts/skill_resources.py` - static learning resources for missing skills.
- `scripts/predict.py` - job category prediction.

## Data

- `job_data_saudi_clean.csv` - job display data used by Streamlit.
- `data/processed/cleaned_data.csv` - cleaned job descriptions used by recommender.
- `data/labels/` - labeling/adjudication files used for classifier training.
- `data/processed/job_embeddings_*.npy` and `job_faiss_*.index` - generated vector cache files.

## Reports And Evaluation

- `reports/recommendation_benchmark.csv` - recommendation benchmark output.
- `reports/model_comparison*.csv` - classifier comparison outputs.
- `reports/confusion_matrix_*` - classifier evaluation files.

## Notebooks And Manual Checks

- `notebooks/` - EDA and manual comparison notebooks.
- `tests/manual/` - manual local test scripts.

## Ignore / Do Not Share

These are local or generated and should not be sent to teammates:

- `.venv/`
- `.claude/`
- `.ipynb_checkpoints/`
- `__pycache__/`
- `models/`
- `saudi_job_market.csv`
- `Job Market Intelligence AI/` nested duplicate folder
