import os
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.getenv("OPENAI_API_VERSION")

PROMPT_TEMPLATE = """
You are a highly skilled AI HR assistant specialized in candidate screening for a Software Engineer role.

Given a candidate's resume and a job description, analyze the following:

1. Extract key skills, years of experience, and qualifications mentioned in the resume.
2. Extract key skills, required experience, and qualifications from the job description.
3. Compare the two and calculate a match percentage score indicating fit.
4. List top 5 matched skills.
5. Identify missing or weak skills.
6. Provide a summary in 5 bullet points (short and specific).
7. Provide a final conclusion: Qualified? Yes/No with explanation.

Job Description:
{job_description}

Candidate Resume:
{resume_text}

Format:
1. Candidate Name: <name>
2. Match Score: <score out of 100>
3. Top 5 Matched Skills: <comma-separated list>
4. Summary:
  • Point 1
  • Point 2
  • ...
5. Conclusion: Qualified? Yes/No with reason in one line
"""