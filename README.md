# HR-Tech: AI-Powered Resume Screening & Employee Engagement Analysis

🚀 A dual-purpose HR toolkit that leverages **AI & ML** for:
- Intelligent **resume screening** based on job descriptions using **GPT-4.1**
- Comprehensive **employee sentiment analysis** to predict **attrition risk** and suggest engagement strategies

## Overview

This project was built as part of an AI/ML internship and addresses two key HR challenges:
1. **Resume Screening** – Automatically evaluate resumes against job descriptions.
2. **Employee Engagement** – Analyze employee feedback to predict attrition and offer retention advice.

---

## 📌 Task 1: Resume Screening

### 🔍 Problem
Manual resume screening is time-consuming and prone to human bias. There’s a need for an AI-driven solution to extract relevant information and evaluate fit based on job descriptions.

### 💡 Solution
An automated pipeline that:
- Accepts a folder of **PDF resumes** and a **.txt job description**
- Uses **Azure OpenAI GPT-4.1** to extract and match:
  - Skills, experience, qualifications
  - Calculates a **match score**
  - Identifies top 5 skills
  - Generates a summary & verdict (Yes/No)
- Outputs a **ranked CSV** of all candidates

### 🛠️ Tools & Techniques
- `pdfplumber` for PDF parsing
- GPT-4.1 structured prompt engineering
- Regex & fallback logic for consistent output

### 📁 Output CSV Fields
- Candidate Name
- Match Score (0–100)
- Top 5 Matched Skills
- Summary (5 bullet points)
- Qualification Verdict (Yes/No with reason)

---

## 📌 Task 2: Employee Sentiment & Attrition Analysis

### 🔍 Problem
Traditional surveys capture feedback but lack deep insights into employee risk and engagement. Qualitative data is underutilized.

### 💡 Solution
A pipeline that:
1. Accepts a **CSV of employee survey data**
2. Performs **sentiment analysis** using **VADER**
3. Predicts **attrition probability** using **XGBoost**
4. Uses GPT-4.1 to generate:
   - Risk level: **High / Medium / Low**
   - **Personalized suggestions** for engagement

### 🧠 Model Components
- VADER (NLTK): For emotion scores (0 to 1 scale)
- XGBoost: For attrition prediction (score scaled 0–10)
- GPT-4.1: For natural language recommendations

### 📁 Output CSV Fields
- Employee ID
- Feedback
- Sentiment Label & Score
- Attrition Score
- Risk Level
- Engagement Suggestion


## Key Features

- Fully **automated resume screening** pipeline
- Actionable **attrition predictions**
- Highly interpretable **LLM outputs**
- **Azure OpenAI** integration
- Clean and structured **CSV outputs**

## Contact

For questions or support, please contact:

**R Sai Shivani** - saishivani0304@gmail.com  
