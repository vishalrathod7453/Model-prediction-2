import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from streamlit_lottie import st_lottie
import requests

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Health Analytics", page_icon="🧪", layout="wide")

# --- CUSTOM CSS FOR ANIMATION & GLASSMORPHISM ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #051937, #004d7a, #008793, #00bf72, #a8eb12);
        background-attachment: fixed;
        color: white;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #051937;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        height: 3.5em;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(146, 254, 157, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FAIL-SAFE LOTTIE LOADER ---
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_health = load_lottie("https://lottie.host/62e08a63-718e-4903-90d0-40e8a716c52b/P9eCshZ2H3.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "Model2.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# --- HEADER ---
st.title("🔬 Smart Health Diagnostic Portal")
st.write("Using High-Precision K-Nearest Neighbors for Predictive Analysis")

if not model:
    st.error("⚠️ File 'Model2.pkl' not found. Please ensure the file is in the same folder as app.py.")
    st.stop()

# --- MAIN INTERFACE ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if lottie_health:
            st_lottie(lottie_health, height=350, key="health_ani")
        else:
            st.title("🏥") # Fallback icon
            
    with col2:
        st.subheader("Patient Vitals Entry")
        c_a, c_b = st.columns(2)
        
        with c_a:
            preg = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose", 0, 300, 100)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
        with c_b:
            ins = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            dpf = st.number_input("Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 1, 120, 25)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 ANALYZE PATIENT DATA"):
            # Prepare data
            input_features = np.array([[preg, glucose, bp, skin, ins, bmi, dpf, age]])
            
            with st.spinner("Classifying data points..."):
                prediction = model.predict(input_features)
                
                st.balloons()
                st.divider()
                
                # Result Display
                res = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
                color = "#ff4b4b" if res == "POSITIVE" else "#00ffcc"
                
                st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 15px;">
                        <h3 style="margin: 0;">PREDICTION RESULT:</h3>
                        <h1 style="color: {color}; margin: 0;">{res}</h1>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
