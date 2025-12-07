from dotenv import load_dotenv

import os
import json

import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

# Load variables from .env into environment
load_dotenv()

# Get API key explicitly
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Check your .env file in this folder.")
    raise RuntimeError("OPENAI_API_KEY is not set")

# Set up OpenAI client using the key
client = OpenAI(api_key=api_key)

# Load ML model (this is a pipeline with preprocessing!)
model = joblib.load('fraud_detection_model2.pkl')

st.title("Credit Card Fraud Detection")
st.markdown("Enter the transaction details to predict if it's fraudulent or not.")
st.divider()

# ---- Input fields ----
# FIXED: Match exact training data values
transaction_type = st.selectbox(
    "Transaction Type", 
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT", "DEBIT"]
)

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance of Origin Account", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance of Origin Account", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance of Destination Account", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance of Destination Account", min_value=0.0, value=0.0)

# Fixed fraud detection threshold (not shown in UI)
fraud_threshold = 0.10


def generate_ai_explanation(features: dict, prediction: int, fraud_prob: float) -> str:
    """
    Call OpenAI to generate a user-friendly explanation of why
    the transaction was (or wasn't) flagged as fraud.
    """

    # Map label to human-readable meaning
    label_text = "fraudulent" if prediction == 1 else "legitimate"

    # Build a compact payload that the model can reason over
    payload = {
        "transaction_features": features,
        "model_output": {
            "prediction_label": int(prediction),
            "prediction_meaning": label_text,
            "fraud_probability": float(fraud_prob)
        }
    }

    system_prompt = (
        "You are an assistant for a fraud detection system. "
        "Your job is to explain to a non-technical user why a transaction was "
        "predicted as potentially fraudulent or legitimate.\n\n"
        "Rules:\n"
        "- Always mention that this is a prediction, not a guarantee.\n"
        "- Focus on the main risk factors only (2-4 reasons).\n"
        "- Be concise: 2-5 sentences.\n"
        "- Use simple language.\n"
        "- Do not invent data that is not in the JSON.\n"
    )

    user_message = (
        "Here is the transaction and model output as JSON. "
        "Explain the main reasons for this prediction in plain English.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"(Could not generate AI explanation: {e})"


if st.button("Predict"):
    # FIXED: Create input with EXACT same columns and order as training
    input_data = pd.DataFrame([{
        'type': transaction_type,  # Must match training exactly (CASH_OUT not "CASH OUT")
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Get fraud probability FIRST (before threshold decision)
    proba = float(model.predict_proba(input_data)[0][1])  # prob of class "1" = fraud
    
    # FIXED: Use custom threshold instead of default 0.5
    prediction = 1 if proba >= fraud_threshold else 0

    # Display results
    st.subheader("Prediction Results")
    
    st.metric("Fraud Probability", f"{proba:.1%}")

    if prediction == 1:
        st.error(f"⚠️ **FRAUD ALERT**: This transaction is predicted to be Fraudulent")
    else:
        st.success(f"✅ **LEGITIMATE**: This transaction appears to be Legitimate")

    # Build a dict of the raw feature values for the AI explanation
    feature_dict = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    st.markdown("### AI Explanation")
    with st.spinner("Generating explanation..."):
        explanation = generate_ai_explanation(feature_dict, prediction, proba)
    st.info(explanation)
    
    # Show debugging info in expander
    with st.expander("🔍 Technical Details"):
        st.write("**Input DataFrame:**")
        st.dataframe(input_data)
        st.write(f"**Raw Prediction:** {prediction}")
        st.write(f"**Fraud Probability:** {proba:.4f}")
        st.write(f"**Applied Threshold:** {fraud_threshold}")