import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def categorize_sentiment(compound_score):
    if compound_score >= 0.6:
        return "Strongly Positive"
    elif compound_score >= 0.2:
        return "Positive"
    elif compound_score > -0.2:
        return "Neutral"
    elif compound_score > -0.6:
        return "Negative"
    else:
        return "Strongly Negative"

def analyze_sentiment(feedback_list):
    sia = SentimentIntensityAnalyzer()
    results = []
    for feedback in feedback_list:
        scores = sia.polarity_scores(str(feedback))
        sentiment_category = categorize_sentiment(scores['compound'])
        results.append({
            'feedback': feedback,
            'compound': scores['compound'],
            'sentiment_label': sentiment_category
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = pd.read_csv("input/survey.csv")  # adjust path & filename
    sentiment_df = analyze_sentiment(df['Feedback'])
    sentiment_df.to_csv("output/feedback_sentiment.csv", index=False)
    print("Sentiment analysis completed. Results saved to output/feedback_sentiment.csv")
