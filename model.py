from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

model_uri = 'runs:/7d2e438570814b2b85d338be304adfe8/model'
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI(  
    title="Target Severity Classification API",
    description="Predict Target Severity Score using MLflow model",
    version="1.0"
)

class PatientData(BaseModel):
    Genetic_Risk: float
    Air_Pollution: float
    Alcohol_Use: float
    Smoking: float
    Obesity_Level: float
    Treatment_Cost_USD: float
    Survival_Years: float

@app.get("/")
def home():
    return {"message": "🚀 Target Severity Prediction API is running!"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
