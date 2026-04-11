import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="HealthPredict AI", page_icon="🏥", layout="wide")

# --- CUSTOM NEON STYLE ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
        color: white;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1 {
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SAFE ANIMATION LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_medical = load_lottieurl("https://lottie.host/62e08a63-718e-4903-90d0-40e8a716c52b/P9eCshZ2H3.json")

# --- ROBUST MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "Model2.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model.pkl, "rb") as f:
        return pickle.load(f)

model = load_model()

# --- HEADER ---
st.title("🏥 HealthPredict Diagnostic Portal")
st.markdown("<p style='text-align: center; color: #4facfe;'>Advanced KNN Analysis for Diabetes Risk Assessment</p>", unsafe_allow_html=True)

if not model:
    st.error("❌ 'Model2.pkl' not found in the directory. Please upload the file to your repository.")
    st.stop()

# --- MAIN CONTENT ---
with st.container():
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        if lottie_medical:
            st_lottie(lottie_medical, height=400)
        else:
            st.markdown("<h1 style='font-size: 150px;'>🩺</h1>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("📋 Patient Metrics")
        
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose Level", 0, 300, 120)
            bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
            skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.number_input("Insulin Level (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age (Years)", 1, 120, 30)
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("RUN DIAGNOSTIC")
        st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION LOGIC ---
if predict_btn:
    # Prepare input exactly as Model2 expects (8 features)
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    with st.status("Analyzing medical markers...", expanded=True) as status:
        st.write("Encoding physiological data...")
        prediction = model.predict(features)
        st.write("Calculating neighbor distance...")
        status.update(label="Diagnostic Ready!", state="complete", expanded=False)

    st.balloons()
    
    # Styled Result
    result_text = "DIABETIC" if prediction[0] == 1 else "NON-DIABETIC"
    result_color = "#ff4b4b" if prediction[0] == 1 else "#00f2fe"
    
    st.markdown(f"""
        <div style="text-align: center; margin-top: 30px; padding: 30px; border-radius: 15px; border: 2px solid {result_color}; background: rgba(255,255,255,0.05);">
            <h3 style="margin: 0; color: #ccc;">Diagnosis Result:</h3>
            <h1 style="color: {result_color}; margin: 0; -webkit-text-fill-color: {result_color};">{result_text}</h1>
        </div>
    """, unsafe_allow_html=True)
