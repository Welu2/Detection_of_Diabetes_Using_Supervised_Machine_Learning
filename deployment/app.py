import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load preprocessing pipeline and model
pipeline = joblib.load("../src/preprocessing_pipeline.pkl")
model = joblib.load("../models/random_forest_model.pkl")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient medical details to predict diabetes risk.")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
	"BMI_Age": bmi * age,
        "Age": age
    }])

    processed_input = pipeline.transform(input_df)
    prediction = model.predict(processed_input)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Diabetes")
    else:
        st.success("✅ Low risk of Diabetes")
