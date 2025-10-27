# ====================================
# XGBoost Model with MLflow
# ====================================

import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# لو y فيها عمود واحد فقط نحولها إلى Series
if y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]

mlflow.set_experiment("Target_Severity_Classification1")

with mlflow.start_run(run_name="XGBoost_Runs"):

    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.2,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',  
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ XGBoost Model trained successfully!")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.2)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.log_param("objective", "multi:softmax")
    mlflow.log_param("num_class", 3)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(xgb_model, "model")

print("\n🎯 XGBoost run logged successfully to MLflow!")
