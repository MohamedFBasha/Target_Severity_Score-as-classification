🎯 Target_Severity_Score (Classification Project)

This machine learning project predicts target severity levels using three models:

Logistic Regression

Random Forest

XGBoost

The project is built with FastAPI, MLflow, and Docker for experiment tracking, deployment, and scalability.

⚙️ Setup

Activate the MLflow environment:

conda activate mlflow_env


Run MLflow UI:

mlflow ui


Open your browser and go to http://127.0.0.1:5000
 to view experiment results.

🧠 Model Tracking

Run each Python file (each represents a different model).
After training, select the best model and set its URI in model.py:

model_uri = 'runs:/d89a184ef59c4827afc62478c3ab07c3/model'

🚀 Run the API

Start FastAPI server:

uvicorn model:app --reload


API will be available at: http://127.0.0.1:8000

📡 Test with Postman

POST → http://127.0.0.1:8000/predict

Body (JSON):

{
  "Genetic_Risk": 0.8634533066302684,
  "Air_Pollution": 0.20291512788939628,
  "Alcohol_Use": 1.4495249647802109,
  "Smoking": -1.696922788884793,
  "Obesity_Level": -0.8969158693733759,
  "Treatment_Cost_USD": 0.6751916180956227,
  "Survival_Years": -0.03668741875825696
}

🧩 Tech Stack

Python

MLflow – Model tracking

FastAPI – API deployment

Docker – Containerization

Scikit-learn, XGBoost
