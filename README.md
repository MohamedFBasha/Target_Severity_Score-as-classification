HAVE 3 PYTHON File run each one in mlflow

#in cmd
conda activate mlflow_env
mlflow ui

Select best model and put model url in model
model_uri = 'runs:/d89a184ef59c4827afc62478c3ab07c3/model'

#in cmd
uvicorn model:app --reload

in postman ----->oost http://127.0.0.1:8000/predict
{
  "Genetic_Risk": 0.8634533066302684,
  "Air_Pollution": 0.20291512788939628,
  "Alcohol_Use": 1.4495249647802109,
  "Smoking": -1.696922788884793,
  "Obesity_Level": -0.8969158693733759,
  "Treatment_Cost_USD": 0.6751916180956227,
  "Survival_Years": -0.03668741875825696
}


