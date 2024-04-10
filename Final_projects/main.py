
import dill

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('data/action_pipe.pkl', 'rb') as file:
    best_model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: float
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    Result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return best_model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict])
    y = best_model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'Result': y[0]
    }
