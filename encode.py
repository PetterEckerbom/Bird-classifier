import base64
import requests
import json

image_path = "allImages/hazgro1/4f7f1c2c4014b4ac3900377a886c765953482e91.jpg"

with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

url = "http://localhost:8080/predict"

payload = {
    "image": encoded_string
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, json=payload, headers=headers)

response_json = response.json()
print(json.dumps(response_json, indent=4))
