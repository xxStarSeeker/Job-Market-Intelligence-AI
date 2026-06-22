import requests

# Prompt exactly as we'll use in the mapping script
MAPPING_PROMPT = """Complete the sentence with a list of skills.

Sentence: "Someone who is proficient with {term} is likely skilled in"

Return ONLY the continuation as a JSON array of 1-4 word skill names. No explanation.

Example:
Sentence: "Someone who is proficient with Apache Spark is likely skilled in"
Output: ["Big Data Processing", "Distributed Computing", "Data Engineering", "ETL", "Scala Programming"]

Example:
Sentence: "Someone who is proficient with AWS is likely skilled in"
Output: ["Cloud Computing", "Cloud Architecture", "Cloud Security", "Infrastructure as Code", "Serverless Computing"]

Example:
Sentence: "Someone who is proficient with CDMP is likely skilled in"
Output: ["Data Governance", "Data Architecture", "Data Quality", "Metadata Management", "Data Stewardship"]

Example:
Sentence: "Someone who is proficient with API testing is likely skilled in"
Output: ["Software Testing", "Test Automation", "API Design", "Quality Assurance", "Integration Testing"]

Example:
Sentence: "Someone who is proficient with AI is likely skilled in"
Output: ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Model Deployment"]

Sentence: "Someone who is proficient with {term} is likely skilled in"
Output:"""

# Test with a few terms that previously failed
test_terms = ["AWS", "AI", "Apache Spark", "CDMP", "API testing"]

for model in ["qwen2.5:32b-instruct-q4_K_M", "qwen2.5:7b-instruct"]:
    print(f"\n=== Model: {model} ===")
    for term in test_terms:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": MAPPING_PROMPT.format(term=term),
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 200},
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["response"]
            print(f"  {term}: {raw[:120]}")
        except Exception as e:
            print(f"  {term}: ERROR - {e}")