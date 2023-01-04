import pandas as pd
import os

from fastapi import FastAPI
from pydantic import BaseModel

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

# Code needed to download DVC data
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()
MODEL = pd.read_pickle('./starter/model/model.pkl')
ENCODER = pd.read_pickle('./starter/model/encoder.pkl')
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class ModelFeatures(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
            }
        }

    def parse_model_features_to_df(self):
        features = {
            "age": [self.age],
            "workclass": [self.workclass],
            "fnlgt": [self.fnlgt],
            "education": [self.education],
            "education-num": [self.education_num],
            "marital-status": [self.marital_status],
            "occupation": [self.occupation],
            "relationship": [self.relationship],
            "race": [self.race],
            "sex": [self.sex],
            "capital-gain": [self.capital_gain],
            "capital-loss": [self.capital_loss],
            "hours-per-week": [self.hours_per_week],
            "native-country": [self.native_country]
        }

        return pd.DataFrame.from_dict(data=features)


@app.post('/predict/')
async def predict(model_features: ModelFeatures):
    features_df = model_features.parse_model_features_to_df()
    processed_features_df, _, _, _ = process_data(X=features_df,
                                                  categorical_features=CAT_FEATURES,
                                                  training=False,
                                                  encoder=ENCODER)
    prediction = inference(MODEL, processed_features_df)
    salary_pred = '<=50K' if prediction[0] == 0 else '>50k'
    return {"salary": salary_pred}


@app.get("/")
async def root():
    return {"message": "Hello World"}
