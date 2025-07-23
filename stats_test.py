import requests

# Make the GET request
response = requests.get('http://localhost:8000/api/statistics')

# Print the response
print(f"Status Code: {response.status_code}")
print("\nCollection Statistics:")
print(response.json())
