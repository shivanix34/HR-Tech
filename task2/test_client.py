import requests

url = "http://127.0.0.1:5060/analyze/"

files = {
    'file': ('survey.csv', open('input/survey.csv', 'rb'), 'text/csv')
}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('output_from_api.csv', 'wb') as f:
        f.write(response.content)
    print("Output saved to output_from_api.csv")
else:
    print("Request failed:", response.status_code, response.text)