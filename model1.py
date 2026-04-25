import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
from scipy.stats import ks_2samp

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("retail_sales_data.csv")

# -----------------------------
# Preprocessing
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'])

df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['weekday'] = df['Date'].dt.weekday

df.drop('Date', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['ProductID', 'Category', 'Region'], drop_first=True)

# -----------------------------
# Train/Test Split
# -----------------------------
X = df.drop('UnitsSold', axis=1)
y = df['UnitsSold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# MLflow Tracking
# -----------------------------
mlflow.set_experiment("Retail_Demand_Forecasting")

with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    print("MAE:", mae)
    print("RMSE:", rmse)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.log_model(model, "model1")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model1.pkl")

# -----------------------------
# Save training columns (VERY IMPORTANT for API)
# -----------------------------
joblib.dump(X_train.columns.tolist(), "columns1.pkl")

# -----------------------------
# Drift Detection Function
# -----------------------------
def detect_drift(train_df, new_df, column):
    stat, p_value = ks_2samp(train_df[column], new_df[column])
    
    if p_value < 0.05:
        return f"Drift detected in {column}"
    else:
        return f"No drift in {column}"


# Example usage
# new_data = pd.read_csv("data/new_data.csv")
# print(detect_drift(df, new_data, "Price"))