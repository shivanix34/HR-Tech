import requests
from script import AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_KEY, AZURE_API_VERSION

def call_azure_openai(prompt):
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are an expert resume screening assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
