from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import os
import shutil
from analysis import (
    sentiment_pipeline, map_sentiment, attrition_model, expected_features,
    generate_suggestion, parse_response
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import uvicorn

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/analyze/")
async def analyze_survey(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(upload_path)

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

    output_file = os.path.join(OUTPUT_DIR, f"analysis_{file.filename}")
    output_cols = ['Employee ID', 'Feedback', 'sentiment_label', 'sentiment_score', 'attrition_score', 'attrition_risk', 'suggestion']
    df.to_csv(output_file, columns=output_cols, index=False)

    return FileResponse(output_file, media_type='text/csv', filename=f"analysis_{file.filename}")


# ✅ Run Uvicorn when executing this script
if __name__ == "__main__":
    uvicorn.run("app_main:app", host="0.0.0.0", port=8000, reload=True)
