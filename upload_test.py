import requests

# File to upload
file_path = 'guide-to-my-health-insurance-coverage-2023.pdf'

# Open the file
with open(file_path, 'rb') as f:
    # Create the multipart form data
    files = {'file': (file_path, f, 'application/pdf')}
    headers = {'accept': 'application/json'}
    
    # Make the POST request
    response = requests.post('http://localhost:8000/api/upload', files=files, headers=headers)
    
    # Print the response
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(response.json())
