import pandas as pd
import numpy as np
import joblib
import os
import textwrap
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from call_api import call_azure_openai

nltk.data.path.append(os.getenv("NLTK_DATA", "/usr/local/nltk_data"))
analyzer = SentimentIntensityAnalyzer()
load_dotenv()

# Load once
attrition_model = joblib.load('models/attrition_model_all_cols.pkl')
expected_features = joblib.load('models/features_used.pkl')


def sentiment_pipeline(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0.0
    score = analyzer.polarity_scores(str(text))
    return score['compound']


def map_sentiment(score):
    if score >= 0.6:
        return "Strongly Positive"
    elif score >= 0.2:
        return "Positive"
    elif score > -0.2:
        return "Neutral"
    elif score > -0.6:
        return "Negative"
    else:
        return "Strongly Negative"


def generate_suggestion(entry):
    prompt = textwrap.dedent(f"""
    You are an expert HR consultant and employee engagement specialist. Based on the employee's feedback, sentiment, and attrition score, do the following:

    1. Assess the employee's attrition risk level as one of these categories: 
        a. High Risk
        b. Medium Risk
        c. Low Risk

    2. Provide a personalized, empathetic, and actionable suggestion to improve retention, engagement, or satisfaction.

    Requirements:
        a. Use clear, supportive, and professional language.
        b. If the feedback is missing, vague, or unclear, rely more on sentiment and attrition score to assess risk and advice.
        c. For neutral or mixed sentiments, carefully consider attrition score to adjust the risk.
        d. Keep the explanation concise (2-3 sentences).
        e. Do NOT repeat the input details in your response.

    Input Details:
        1. Employee Feedback: {entry.get('Feedback', 'No feedback provided')}
        2. Sentiment Label: {entry.get('sentiment_label', 'N/A')}
        3. Sentiment Score (0 to 1): {entry.get('sentiment_score', 'N/A')}
        4. Attrition Score (0 to 10): {entry.get('attrition_score', 'N/A')}

    Output Format:
    Attrition Risk: <High Risk / Medium Risk / Low Risk>
    Suggestion: <Your concise, actionable recommendation>

    Output:
    """)
    try:
        suggestion = call_azure_openai(prompt)
    except Exception:
        suggestion = "Suggestion could not be generated due to API error."
    return suggestion if suggestion else "Suggestion could not be generated."


def parse_response(text):
    lines = text.split('\n')
    risk = ''
    suggestion = ''
    for line in lines:
        if line.strip().startswith('Attrition Risk:'):
            risk = line.split(':', 1)[1].strip()
        elif line.strip().startswith('Suggestion:'):
            suggestion = line.split(':', 1)[1].strip()
    if not risk:
        risk = "Unknown"
    if not suggestion:
        suggestion = text.strip()
    return risk, suggestion


def analyze_df(df: pd.DataFrame) -> pd.DataFrame:
    df['sentiment_score_raw'] = df['Feedback'].apply(sentiment_pipeline)
    df['sentiment_label'] = df['sentiment_score_raw'].apply(map_sentiment)
    df['sentiment_score'] = ((df['sentiment_score_raw'] + 1) / 2).round(2)

    X = df.copy()
    drop_cols = ['Feedback', 'Attrition', 'sentiment_score', 'sentiment_score_raw', 'sentiment_label']
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    X = X.loc[:, X.columns.intersection(expected_features)]

    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    try:
        attrition_probs = attrition_model.predict_proba(X)[:, 1]
        df['attrition_score'] = (attrition_probs * 10).round(2)
    except Exception:
        df['attrition_score'] = (np.random.rand(len(df)) * 10).round(2)

    records = df.to_dict(orient='records')
    raw_suggestions = [generate_suggestion(r) for r in records]

    risks = []
    clean_suggestions = []
    for s in raw_suggestions:
        risk, sugg = parse_response(s)
        risks.append(risk)
        clean_suggestions.append(sugg)

    df['attrition_risk'] = risks
    df['suggestion'] = clean_suggestions

    return df
