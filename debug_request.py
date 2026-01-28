import requests
import os

url = 'http://127.0.0.1:5001/predict'
img_path = "/Users/tanishq/.gemini/antigravity/brain/a4bbae1e-0fb8-483b-92a1-8923705c5f88/uploaded_media_0_1769622986432.png"

print(f"Sending POST request to {url} with image: {img_path}")

try:
    with open(img_path, 'rb') as f:
        files = {'file': f}
        r = requests.post(url, files=files)
        
    print(f"Status Code: {r.status_code}")
    print(f"Response Body: {r.text}")
except Exception as e:
    print(f"Request Error: {e}")
