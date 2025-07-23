import requests

# Test query
query = "What are my pre-existing condition coverage limits?"

# Prepare request
headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
data = {
    'query': query,
    'use_llm': True,
    'top_k': 5
}

# Make the query
response = requests.post('http://localhost:8000/api/query', json=data, headers=headers)

# Print response
print(f"Status Code: {response.status_code}")
print("\nResponse:")
print(response.json())
