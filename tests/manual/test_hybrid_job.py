"""Compare classification of the hybrid Data Analyst / Analytics Engineer job."""
import sys, json, requests
from pathlib import Path

# Allow importing from our own project
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.benchmark_classifiers import clean_text

# The raw job description (copy-pasted from earlier)
RAW_DESC = r"""About the job
We're looking for a Senior Data Analyst/ Analytics Engineer to own data and analytics across our Gen AI and Recommendation Systems work. It's a hybrid role: you'll own the centralized reporting that turns data into decisions and build the pipelines and data models that feed it — defining the right metrics for each product we ship rather than waiting on others to prepare your data. For Recommendation Systems, you'll bring enough ML understanding to engineer the right features and evaluation metrics, partnering closely with Data Scientists, ML Engineers, Product, and Backend teams.

Key Responsibilities

Pipeline Architecture & Development: Build and maintain scalable, fault-tolerant batch and streaming pipelines that serve analytical and ML use cases
Centralized Reporting & Metrics: Define the key metrics for each product we ship and build rock-solid centralized reporting around them, surfacing the trends and insights that matter
Data Modeling: Design and own multi-layer data models (staging to feature-ready marts) that stay consistent and performant across ML models, dashboards, and APIs, handling schema changes cleanly
Feature Store & ML Data Flows: Engineer the data flows that populate and update our ML Feature Store (and graph data where relevant) with the availability and low latency recommendation models need
Experimentation & A/B Testing: Build the pipelines and metrics frameworks behind A/B testing — experiment schemas, assignment logging, and reliable metric computation for statistically sound results
ClickHouse Mastery: Own ClickHouse as the domain expert — schema design, performance tuning, and fast queries for experiment aggregation and feature serving
Streaming & CDC: Implement Change Data Capture (CDC) and event-driven flows (e.g. Apache Kafka) to keep data fresh where reporting and recommendations need it
Orchestration & Automation: Build and manage workflows with modern orchestration tools (e.g. Mage AI, Airflow, Prefect) for reliable delivery and dependency management
ML-Aware Support: Define and interpret the right offline and online ranking metrics, and engineer the features the models actually need
Cross-Functional Collaboration: Partner with Data Scientists, ML Engineers, Product, and Backend to turn data requirements into production pipelines and actionable ML features

Requirements

Experience: 4+ years as a Data/Analytics Engineer building data systems for analytics and ML
Programming: Expert Python and advanced SQL
BI & Visualization: Strong BI/visualization skills (e.g. Looker, Tableau) and good intuition for which metrics matter and how to present them
Pipelines & Orchestration: Hands-on building pipelines with modern orchestration (Mage AI, Airflow, Prefect) — you build your own data, not just consume it
Data Warehouse / ClickHouse: Deep production experience with ClickHouse (or BigQuery, Snowflake, or similar)
Data Modeling: Hands-on multi-layer modeling (raw, staging, marts) using Kimball, Data Vault, or OBT patterns
Experimentation & A/B Testing: Solid grasp of experimentation frameworks — assignment, holdouts, metric pipelines, variance reduction
ML Exposure: Good grasp of the ML lifecycle — how models consume data, how Feature Stores work (e.g. Feast, Hopsworks), and how to engineer features at scale, plus enough ranking-metric knowledge to support Recommendation Systems

Nice to have:

DBT for modeling and transformation
Building or integrating A/B platforms (e.g. Statsig, Optimizely, GrowthBook, or custom)
Apache Kafka and CDC tools (e.g. Debezium, Maxwell)
Graph Databases (e.g. Dgraph, Neo4j, Amazon Neptune) and structuring data for them
JavaScript or Go"""
# ---------------------------------------------------------------------------
# 3. Main – clean the text and send to both classifiers
# ---------------------------------------------------------------------------
def main():
    cleaned = robust_clean(RAW_DESC)

    # Extract a short title from the first line (optional, but their schema requires it)
    title = cleaned.split('.')[0].strip()[:200]   # first sentence, capped at 200 chars

    # Truncate description to avoid length limits (their config may restrict to 600 chars, so be safe)
    short_desc = cleaned[:5000]

    print("=== CLEANED TEXT (first 300 chars) ===")
    print(cleaned[:300] + "...\n")
    print("=== EXTRACTED TITLE ===")
    print(title + "\n")

    # ---- Your classifier (port 8001) ----
    print("=== YOUR CLASSIFIER (port 8001) ===")
    try:
        resp = requests.post(
            "http://127.0.0.1:8001/classify",
            json={"description": cleaned},
            timeout=30,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

    # ---- Teammate classifier (port 8000) ----
    print("\n=== TEAMMATE CLASSIFIER (port 8000) ===")
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/classify",
            json={
                "job_title": title,
                "job_description": short_desc,
            },
            timeout=30,
        )
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()