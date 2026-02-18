import requests
import json

# API endpoint URL
url = "http://localhost:5000/predict"

# Test data
test_data = {
    "temperature": 23.5,
    "humidity": 68.2,
    "gas_resistance": 95000,
    "iaq": 108
}

# Send request
response = requests.post(url, json=test_data)

# Display result
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.2%}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)