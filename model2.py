import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("machine_sensor_data.csv")

# -----------------------------
# Preprocessing
# -----------------------------
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['hour'] = df['Timestamp'].dt.hour
df['day'] = df['Timestamp'].dt.day

df.drop('Timestamp', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['MachineID'], drop_first=True)

# -----------------------------
# Split
# -----------------------------
X = df.drop('Failure', axis=1)
y = df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# MLflow Tracking
# -----------------------------
mlflow.set_experiment("Predictive_Maintenance")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model2.pkl")
joblib.dump(X_train.columns.tolist(), "columns2.pkl")