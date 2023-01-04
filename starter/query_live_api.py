import requests

url = "https://scalable-fastapi-model.herokuapp.com/predict/"
features = {
    "age": 59,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Divorced",
    "occupation": "Adm-clerical",
    "relationship": "Wife",
    "race": "White",
    "sex": "Male",
    "capital_gain": 12174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post(url, json=features)
print(response.status_code)
print(response.text)

