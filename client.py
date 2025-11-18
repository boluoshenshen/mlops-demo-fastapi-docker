import requests

data = {
    "features": [0.5, -1.2, 0.3, 0.9, -0.4]
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())