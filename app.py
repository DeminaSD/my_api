from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionInput(BaseModel):
    Stress_Level: int
    Job_Satisfaction: int

@app.post("/predict_model")
def predict_data(input_data: PredictionInput):
    new_data = [[input_data.Stress_Level, input_data.Job_Satisfaction]]
    result = model.predict(new_data)[0]
    return {"prediction": int(result)}

import uvicorn
uvicorn.run(app, host = '127.0.0.1', port = 5000)