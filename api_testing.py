import time
import requests

base_url = 'http://116.202.111.229:8000'
api_key = 'api-key'

headers = {
    'x-api-key': ''
}


# Get a new hint for current company or get the first hint for a new company after calling /evaluate/reset
response = requests.get(f"{base_url}/evaluate/hint", headers=headers)

print(response.status_code, response.json())

# predict based off given hint
time.sleep(1)


# Post your answer for current hint
data = {
    'answer': 'Ambulatory Health Care Services'
}
response = requests.post(f"{base_url}/evaluate/answer", json=data, headers=headers)

print(response.status_code, response.json())


# Get hints about a new company
response = requests.get(f"{base_url}/evaluate/reset", headers=headers)

print(response.status_code, response.json())