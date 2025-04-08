import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("random_forest_model.pkl")
try:
    label_encoder = joblib.load("label_encoder.pkl")
except:
    label_encoder = None

st.set_page_config(page_title="ASD Detection App", layout="centered")
st.title("🧠 Autism Spectrum Disorder (ASD) Detection")
st.markdown("Please fill out the form below to assess the likelihood of ASD.")

# User input form
with st.form("asd_form"):
    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["male", "female"])
    jaundice = st.selectbox("Jaundice at birth", ["yes", "no"])
    family_history = st.selectbox("Family history of ASD", ["yes", "no"])
    who_completed = st.selectbox("Who completed the test?", ["self", "parent", "guardian", "other"])

    st.markdown("### AQ-10 Screening Questions")
    aq_responses = []
    for i in range(1, 11):
        response = st.selectbox(f"Q{i}: Answer", ["yes", "no"], key=f"aq{i}")
        aq_responses.append(1 if response == "yes" else 0)

    submitted = st.form_submit_button("Predict ASD")

# Preprocess and predict
if submitted:
    input_data = {
        "A1_Score": aq_responses[0],
        "A2_Score": aq_responses[1],
        "A3_Score": aq_responses[2],
        "A4_Score": aq_responses[3],
        "A5_Score": aq_responses[4],
        "A6_Score": aq_responses[5],
        "A7_Score": aq_responses[6],
        "A8_Score": aq_responses[7],
        "A9_Score": aq_responses[8],
        "A10_Score": aq_responses[9],
        "age": age,
        "gender": 1 if gender == "male" else 0,
        "jaundice": 1 if jaundice == "yes" else 0,
        "family_history": 1 if family_history == "yes" else 0,
        "who_completed_the_test": hash(who_completed) % 100
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    if label_encoder:
        prediction_label = label_encoder.inverse_transform([prediction])[0]
    else:
        prediction_label = "Yes" if prediction == 1 else "No"

    st.subheader("🧾 Prediction Result")
    st.success(f"ASD Traits Detected: **{prediction_label}**")
