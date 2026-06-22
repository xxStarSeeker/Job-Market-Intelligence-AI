import json, requests

# Load both models' results
with open("data/processed/skill_mappings_7b.jsonl", encoding="utf-8") as f:
    data7b = {json.loads(line)["term"]: json.loads(line).get("implied_skills", []) for line in f}
with open("data/processed/skill_mappings_32b.jsonl", encoding="utf-8") as f:
    data32b = {json.loads(line)["term"]: json.loads(line).get("implied_skills", []) for line in f}
# Terms that previously had issues or are critical for demo
test_terms = [
    "CA", "Chatbots", "DAX", "CompTIA Security+", "CCNA", "PMP", "PRINCE2",
    "CISSP", "CloudFormation", "BPMN", "CAP", "CRISC", "Alibaba Cloud"
]

JUDGE_PROMPT = """You are a technical skills expert. Given a tool or certification and two suggested lists of implied skills from different models, produce the single best consolidated list.

Step 1 — Reason about which skills are genuinely implied by the term. Consider:
- Are any skills in the lists irrelevant or misattributed? Remove them.
- Are any important skills missing from both lists? If so, add them.
- Are there near-duplicates? Prefer the most specific and widely recognized canonical name.

Step 2 — Output your final consolidated list.

Return ONLY a JSON object with:
{{
  "consolidated_skills": ["skill1", "skill2", ...],
  "reasoning": "<one short sentence explaining key decisions>",
  "certainty": "high|medium|low"
}}

Term: {term}
List A: {list_a}
List B: {list_b}

JSON response:"""

for term in test_terms:
    list_a = data7b.get(term, [])
    list_b = data32b.get(term, [])
    print(f"\n=== {term} ===")
    print(f"  7B  : {list_a}")
    print(f"  32B : {list_b}")

    prompt = JUDGE_PROMPT.format(term=term, list_a=json.dumps(list_a), list_b=json.dumps(list_b))
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:32b-instruct-q4_K_M",
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 300}
            },
            timeout=30
        ).json()
        result = json.loads(resp["response"])
        print(f"  Judge: {result.get('consolidated_skills')}")
        print(f"  Reason: {result.get('reasoning')} (certainty: {result.get('certainty')})")
    except Exception as e:
        print(f"  Judge failed: {e}")