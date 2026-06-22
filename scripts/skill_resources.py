"""
Static catalogue of one high-quality, FREE learning resource per skill.

Used by the resume optimizer for "Category B — skills to learn": for a missing
skill with no evidence in the candidate's resume, we point them at a concrete
free course/doc instead of asking an LLM at runtime (deterministic, cheap,
consistent). Keys are canonical names from scripts.skill_vocab.SKILLS_VOCABULARY;
look-ups should be case-insensitive. Skills absent here get a generic fallback.

    from scripts.skill_resources import get_skill_resources
"""

from __future__ import annotations

SKILL_RESOURCES: dict[str, str] = {
    # --- Programming languages -------------------------------------------------
    "Python": "Python official tutorial — https://docs.python.org/3/tutorial/",
    "Java": "Java Programming MOOC (Univ. of Helsinki) — https://java-programming.mooc.fi/",
    "JavaScript": "freeCodeCamp JavaScript Algorithms — https://www.freecodecamp.org/learn/",
    "TypeScript": "TypeScript Handbook — https://www.typescriptlang.org/docs/handbook/intro.html",
    "Scala": "Scala official tour — https://docs.scala-lang.org/tour/tour-of-scala.html",
    "Go": "A Tour of Go — https://go.dev/tour/",
    "Bash": "Bash scripting tutorial (Ryan's) — https://ryanstutorials.net/bash-scripting-tutorial/",
    "PHP": "PHP The Right Way — https://phptherightway.com/",
    "Ruby": "Ruby in Twenty Minutes — https://www.ruby-lang.org/en/documentation/quickstart/",
    "C++": "learncpp.com — https://www.learncpp.com/",
    # --- Data / SQL ------------------------------------------------------------
    "SQL": "SQLBolt interactive lessons — https://sqlbolt.com/",
    "PostgreSQL": "PostgreSQL Tutorial — https://www.postgresqltutorial.com/",
    "MySQL": "MySQL Tutorial — https://www.mysqltutorial.org/",
    "MongoDB": "MongoDB University M001 (free) — https://learn.mongodb.com/",
    "Redis": "Redis University (free) — https://university.redis.com/",
    "Elasticsearch": "Elastic free training — https://www.elastic.co/training/free",
    "Pandas": "pandas 'Getting started' — https://pandas.pydata.org/docs/getting_started/",
    "NumPy": "NumPy: the absolute basics — https://numpy.org/doc/stable/user/absolute_beginners.html",
    "matplotlib": "Matplotlib tutorials — https://matplotlib.org/stable/tutorials/",
    "Data Modeling": "Kimball dimensional modeling techniques — https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/",
    "data visualization": "Google Data Studio / Data Viz course — https://www.cloudskillsboost.google/",
    "statistical analysis": "Khan Academy Statistics & Probability — https://www.khanacademy.org/math/statistics-probability",
    # --- Data engineering ------------------------------------------------------
    "Data Engineering": "DataTalks.Club Data Engineering Zoomcamp — https://github.com/DataTalksClub/data-engineering-zoomcamp",
    "ETL": "Airbyte 'ETL pipeline' guides — https://airbyte.com/data-engineering-resources",
    "Apache Spark": "Spark: The Definitive Guide / docs — https://spark.apache.org/docs/latest/quick-start.html",
    "Apache Kafka": "Confluent Kafka 101 (free) — https://developer.confluent.io/learn-kafka/",
    "Apache Airflow": "Airflow tutorial — https://airflow.apache.org/docs/apache-airflow/stable/tutorial/",
    "Data Warehouse": "dbt 'Analytics Engineering' fundamentals (free) — https://learn.getdbt.com/",
    "Business Intelligence": "Microsoft Learn — Power BI — https://learn.microsoft.com/training/powerplatform/power-bi",
    "Power BI": "Microsoft Learn — Power BI — https://learn.microsoft.com/training/powerplatform/power-bi",
    "Tableau": "Tableau free training videos — https://www.tableau.com/learn/training/elearning",
    "data governance": "DAMA-DMBOK overview (free summary) — https://www.dataversity.net/data-governance-fundamentals/",
    "Data Quality": "Great Expectations docs — https://docs.greatexpectations.io/docs/",
    "Metadata Management": "OpenMetadata docs — https://docs.open-metadata.org/",
    # --- ML / AI ---------------------------------------------------------------
    "Machine Learning": "Google ML Crash Course — https://developers.google.com/machine-learning/crash-course",
    "Deep Learning": "fast.ai Practical Deep Learning — https://course.fast.ai/",
    "Data Science": "Kaggle Learn micro-courses — https://www.kaggle.com/learn",
    "feature engineering": "Kaggle Learn — Feature Engineering — https://www.kaggle.com/learn/feature-engineering",
    "NLP": "Hugging Face NLP Course — https://huggingface.co/learn/nlp-course",
    "LLM": "Hugging Face LLM Course — https://huggingface.co/learn/llm-course",
    "Generative AI": "Google 'Intro to Generative AI' (free) — https://www.cloudskillsboost.google/course_templates/536",
    "prompt engineering": "Anthropic prompt engineering docs — https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview",
    "computer vision": "PyImageSearch University free tier — https://pyimagesearch.com/pyimagesearch-university/",
    "Transformers": "Hugging Face Transformers course — https://huggingface.co/learn/nlp-course/chapter1",
    "Hugging Face": "Hugging Face Learn hub — https://huggingface.co/learn",
    "RAG": "LangChain RAG tutorial — https://python.langchain.com/docs/tutorials/rag/",
    "vector databases": "Pinecone 'Vector DB' learning center — https://www.pinecone.io/learn/",
    "MLOps": "DataTalks.Club MLOps Zoomcamp — https://github.com/DataTalksClub/mlops-zoomcamp",
    "TensorFlow": "TensorFlow tutorials — https://www.tensorflow.org/tutorials",
    "PyTorch": "PyTorch official tutorials — https://pytorch.org/tutorials/",
    "Keras": "Keras developer guides — https://keras.io/guides/",
    "scikit-learn": "scikit-learn user guide — https://scikit-learn.org/stable/user_guide.html",
    "AI/ML": "Elements of AI (free) — https://www.elementsofai.com/",
    # --- Cloud / DevOps --------------------------------------------------------
    "AWS": "AWS Cloud Practitioner Essentials (free) — https://aws.amazon.com/training/digital/",
    "AWS SageMaker": "AWS SageMaker — Get Started — https://aws.amazon.com/sagemaker/getting-started/",
    "SageMaker": "AWS SageMaker — Get Started — https://aws.amazon.com/sagemaker/getting-started/",
    "Azure": "Microsoft Learn — Azure Fundamentals — https://learn.microsoft.com/training/paths/azure-fundamentals/",
    "Azure AI": "Microsoft Learn — Azure AI Fundamentals — https://learn.microsoft.com/training/paths/get-started-with-artificial-intelligence-on-azure/",
    "GCP": "Google Cloud Skills Boost (free tier) — https://www.cloudskillsboost.google/",
    "Docker": "Docker 'Get Started' — https://docs.docker.com/get-started/",
    "Kubernetes": "Kubernetes Basics tutorial — https://kubernetes.io/docs/tutorials/kubernetes-basics/",
    "Terraform": "HashiCorp Learn Terraform — https://developer.hashicorp.com/terraform/tutorials",
    "CI/CD": "GitHub Actions documentation — https://docs.github.com/actions",
    "Git": "Pro Git book (free) — https://git-scm.com/book",
    "Linux": "Linux Journey — https://linuxjourney.com/",
    "Jenkins": "Jenkins handbook — https://www.jenkins.io/doc/book/",
    "Ansible": "Ansible 'Getting started' — https://docs.ansible.com/ansible/latest/getting_started/",
    "Prometheus": "Prometheus 'Getting started' — https://prometheus.io/docs/prometheus/latest/getting_started/",
    "Grafana": "Grafana fundamentals (free) — https://grafana.com/tutorials/",
    # --- Web --------------------------------------------------------------------
    "React": "react.dev official tutorial — https://react.dev/learn",
    "Angular": "Angular tutorial — https://angular.dev/tutorials",
    "Node.js": "Node.js 'Get started' guide — https://nodejs.org/en/learn",
    "REST API": "MDN HTTP & REST basics — https://developer.mozilla.org/en-US/docs/Web/HTTP",
    "FastAPI": "FastAPI tutorial — https://fastapi.tiangolo.com/tutorial/",
    "HTML": "MDN HTML basics — https://developer.mozilla.org/en-US/docs/Learn/HTML",
    "CSS": "MDN CSS first steps — https://developer.mozilla.org/en-US/docs/Learn/CSS",
    # --- Security ---------------------------------------------------------------
    "cybersecurity": "TryHackMe Pre-Security (free path) — https://tryhackme.com/path/outline/presecurity",
    "Penetration Testing": "TryHackMe Jr Penetration Tester — https://tryhackme.com/path/outline/jrpenetrationtester",
    "network security": "Professor Messer Security+ (free) — https://www.professormesser.com/security-plus/",
    "SIEM": "Splunk free fundamentals training — https://www.splunk.com/en_us/training/free-courses.html",
    "Information Security": "Professor Messer Security+ (free) — https://www.professormesser.com/security-plus/",
    "Risk Management": "NIST Risk Management Framework overview — https://csrc.nist.gov/projects/risk-management",
    "ISO 27001": "ISO/IEC 27001 overview — https://www.iso.org/isoiec-27001-information-security.html",
    "networking": "Cisco 'Networking Basics' (free) — https://www.netacad.com/courses/networking-basics",
    # --- QA / testing -----------------------------------------------------------
    "Selenium": "Selenium official docs — https://www.selenium.dev/documentation/",
    "Cypress": "Cypress 'Real World' learning — https://learn.cypress.io/",
    "quality assurance": "Ministry of Testing — free resources — https://www.ministryoftesting.com/",
    "test automation": "Test Automation University (free) — https://testautomationu.applitools.com/",
    # --- PM / BA ----------------------------------------------------------------
    "Agile": "Atlassian Agile Coach — https://www.atlassian.com/agile",
    "Scrum": "The Scrum Guide (free) — https://scrumguides.org/",
    "project management": "Google Project Management (Coursera, audit free) — https://www.coursera.org/professional-certificates/google-project-management",
    "stakeholder management": "MindTools stakeholder management — https://www.mindtools.com/aol0rms/stakeholder-management",
    "Business Analysis": "BABOK guide overview / IIBA resources — https://www.iiba.org/career-resources/",
    "JIRA": "Atlassian University — Jira (free) — https://university.atlassian.com/",
}

_GENERIC = ("No curated resource yet — search freeCodeCamp, the official docs, "
            "a Coursera audit, or a YouTube crash course for this skill.")


def get_skill_resources() -> dict[str, str]:
    """Return the static {skill: free-resource} catalogue (copy)."""
    return dict(SKILL_RESOURCES)


def resource_for(skill: str) -> str:
    """One free resource for `skill` (case-insensitive); generic if unknown."""
    if skill in SKILL_RESOURCES:
        return SKILL_RESOURCES[skill]
    lower = {k.lower(): v for k, v in SKILL_RESOURCES.items()}
    return lower.get(skill.lower(), _GENERIC)
