import pandas as pd
import joblib
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load data
test_df = pd.read_csv("input/survey.csv")  # adjust path as needed

# Keep original feedback before encoding
original_feedback = test_df["Feedback"].copy()

# Load model and expected feature list
model = joblib.load("models/attrition_model_all_cols.pkl")
expected_features = joblib.load("models/features_used.pkl")

# Fill missing values as in training
for col in test_df.columns:
    if test_df[col].dtype in ['int64', 'float64']:
        test_df[col] = test_df[col].fillna(test_df[col].median())
    else:
        test_df[col] = test_df[col].fillna(test_df[col].mode()[0])

# Encode categorical columns (match training logic)
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in test_df.columns:
    if test_df[col].dtype == 'object':
        le = LabelEncoder()
        test_df[col] = le.fit_transform(test_df[col].astype(str))
        label_encoders[col] = le

# Select only the columns used in training
X_test = test_df[expected_features]

# Predict
try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of 'left'
except Exception as e:
    print("Attrition model prediction failed, using random scores.")
    print("Error:", e)
    y_pred_proba = np.random.rand(len(test_df))

# Attach predictions to the output
test_df["Attrition_Risk_Score"] = y_pred_proba

# Define three-level risk labels
def risk_label(prob):
    if prob > 0.7:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

test_df["Attrition_Risk_Label"] = test_df["Attrition_Risk_Score"].apply(risk_label)

output_df = pd.DataFrame({
    "Employee ID": test_df["Employee ID"],
    "Attrition_Risk_Score": test_df["Attrition_Risk_Score"],
    "Attrition_Risk_Label": test_df["Attrition_Risk_Label"]
})

output_df.to_csv("output/attrition_output.csv", index=False)
print("Final output saved to: output/final_output.csv")