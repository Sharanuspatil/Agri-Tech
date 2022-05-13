import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'state_name':'Andaman and Nicobar Islands', 'season':'Whole Year', 'crop':'Arecanut','area':'1000.00'})

print(r.json())