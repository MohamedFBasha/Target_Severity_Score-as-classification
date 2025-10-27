FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn mlflow pandas scikit-learn

EXPOSE 8000

CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000"]
