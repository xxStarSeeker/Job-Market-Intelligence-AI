import requests, json

# Minimal test — only 100 chars
short = "We need a Data Engineer to build ETL pipelines with Apache Spark and Airflow."

for field in ["description", "job_description"]:
    for length in [100, 500, 1000, 5000]:
        payload = {
            "job_title": "Data Engineer",
            field: short[:length]
        }
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/classify",
                json=payload,
                timeout=10
            )
            print(f"field={field}, len={length}: {resp.status_code}")
            if resp.status_code == 200:
                print("  SUCCESS:", json.dumps(resp.json(), indent=2))
                break
        except Exception as e:
            print(f"field={field}, len={length}: {e}")
    else:
        continue
    break