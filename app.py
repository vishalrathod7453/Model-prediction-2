import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🏥", layout="wide")

# Custom CSS for a clean medical UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ANIMATION ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_health = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_57pk9mow.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure 'Model2.pkl' is in the same folder as this script on GitHub
    with open('Model2.pkl', 'wb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'Model2.pkl' is uploaded to GitHub.")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Clinical Data")
with st.sidebar:
    preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
    bp = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
    skin = st.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# --- MAIN PAGE FRONTEND ---
st.title("🏥 Diabetes Health Risk Analysis")
st.write("Using AI to evaluate patient health indicators based on the KNN model.")

col1, col2 = st.columns([1, 1])

with col1:
    if lottie_health:
        st_lottie(lottie_health, height=300)

with col2:
    st.markdown("### Analysis Summary")
    st.write("Check the values in the sidebar and click the button below.")
    
    if st.button("Generate Prediction"):
        # The 8 features MUST be in this exact order 
        features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        
        prediction = model.predict(features)
        
        st.divider()
        if prediction[0] == 1:
            st.error("### Result: High Risk Detected")
            st.write("The model suggests a high probability of diabetes. Please consult a medical professional.")
        else:
            st.success("### Result: Low Risk Detected")
            st.balloons()
            st.write("The model suggests a low probability of diabetes based on these parameters.")

st.caption("Disclaimer: This tool is for educational purposes and is not a medical diagnosis.")
