from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)


def test_get_hello_world():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World"}


def test_post_predict_less_than_50k():
    r = client.post("/predict/", json={
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Wife",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"salary": '<=50K'}


def test_post_predict_more_than_50k():
    r = client.post("/predict/", json={
        "age": 39,
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
    })
    assert r.status_code == 200
    assert r.json() == {"salary": '>50k'}
