import requests


API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer hf_<token>"}


def query(payload):
  response = requests.post(API_URL, headers=headers, json=payload)
  return response.json()
 
output = query({
  "inputs": "Can you please let us know more details about your ",
})
