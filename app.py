import os
import joblib
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv

# Import your custom modules
from azure_client import get_ai_response
from services import get_customer_data, create_ticket

# Load environment variables
load_dotenv()

app = FastAPI(title="Unified Retail & Chatbot API")

# -----------------------------
# Initialization
# -----------------------------
# Load ML models and columns
model1 = joblib.load("model1.pkl")
columns1 = joblib.load("columns1.pkl")
model2 = joblib.load("model2.pkl")
columns2 = joblib.load("columns2.pkl")

# Load Intent CSV for the Chatbot
df_intents = pd.read_csv("intents.csv")

def find_intent(user_query):
    """Simple keyword-based intent matching."""
    for _, row in df_intents.iterrows():
        if row["Query"].lower() in user_query.lower():
            return row["Intent"], row["Response"]
    return None, None

# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/chat")
def chat(query: str):
    """Chatbot endpoint: Checks intents first, falls back to Azure AI."""
    intent, response = find_intent(query)

    if intent:
        if intent == "Order_Status":
            customer = get_customer_data()
            return {
                "intent": intent,
                "response": f"{response} (Status: {customer['order_status']})"
            }
        elif intent in ["Cancel_Order", "Payment_Issue"]:
            ticket = create_ticket(intent.replace("_", " "))
            return {
                "intent": intent,
                "response": response,
                "ticket": ticket
            }
        elif intent == "Refund_Query":
            return {"intent": intent, "response": response}

    # Fallback to Azure AI if no intent matched
    ai_response = get_ai_response(query)
    return {"intent": "AI_Fallback", "response": ai_response}

@app.post("/predict-demand")
def predict(data: dict):
    """Retail Demand Forecast."""
    df = pd.DataFrame([data])
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.weekday
    df.drop('Date', axis=1, inplace=True)
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns1, fill_value=0)
    
    prediction = model1.predict(df)
    return {"predicted_demand": int(prediction[0])}

@app.post("/predict-failure")
def predict_failure(data: dict):
    """Failure Probability Prediction."""
    df = pd.DataFrame([data])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df.drop('Timestamp', axis=1, inplace=True)
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns2, fill_value=0)

    prob = model2.predict_proba(df)[0][1]
    return {"failure_probability": float(prob)}