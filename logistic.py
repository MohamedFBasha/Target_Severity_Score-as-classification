# ====================================
# Logistic Regression with MLflow
# ====================================

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ تحميل البيانات
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# لو y فيها عمود واحد فقط، نحوله إلى Series
if y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]

# 2️⃣ إعداد MLflow experiment
mlflow.set_experiment("Target_Severity_Classification1")

with mlflow.start_run(run_name="Logistic_Regression_Runs"):

    # 3️⃣ إنشاء وتدريب النموذج
    log_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    log_model.fit(X_train, y_train)

    # 4️⃣ التنبؤ والتقييم
    y_pred = log_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("✅ Model trained successfully!")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 5️⃣ تسجيل النتائج في MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("multi_class", "multinomial")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", acc)

    # 6️⃣ حفظ النموذج داخل MLflow
    mlflow.sklearn.log_model(log_model, "model")

print("\n🎯 Run logged successfully to MLflow!")
