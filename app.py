import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# 1. Page Config
st.set_config(page_title="AI Usage Predictor", layout="centered")

# 2. Animations
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# 3. Load Model [cite: 1, 85]
@st.cache_resource
def load_model():
    with open("Model1.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# 4. Frontend Styling
st.title("🤖 Smart AI Predictor")
st_lottie(lottie_coding, height=200)

# 5. User Inputs
with st.expander("Enter User Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 80, 25)
        # Using lists derived from your model's feature names 
        gender = st.selectbox("Gender", ["Male", "Female"])
        edu = st.selectbox("Education", ["High School", "Undergrad", "Postgrad"])
        city = st.selectbox("City Category", ["Tier 1", "Tier 2", "Tier 3"])
    with col2:
        tool = st.selectbox("AI Tool", ["ChatGPT", "Gemini", "Claude", "Other"])
        hours = st.slider("Daily Usage", 0.0, 24.0, 3.0)
        purpose = st.selectbox("Primary Purpose", ["Work", "Study", "Creative"])

# 6. Simple Encoder (Match this to your training LabelEncoders!)
mapping = {
    "Male": 0, "Female": 1,
    "High School": 0, "Undergrad": 1, "Postgrad": 2,
    "Tier 1": 0, "Tier 2": 1, "Tier 3": 2,
    "ChatGPT": 0, "Gemini": 1, "Claude": 2, "Other": 3,
    "Work": 0, "Study": 1, "Creative": 2
}

# 7. Prediction Logic
if st.button("Analyze Data"):
    # Transform categorical text to numerical values 
    features = np.array([[
        age, 
        mapping[gender], 
        mapping[edu], 
        mapping[city], 
        mapping[tool], 
        hours, 
        mapping[purpose]
    ]])
    
    prediction = model.predict(features)
    
    st.success(f"### Predicted Class: {prediction[0]}")
    st.balloons()
