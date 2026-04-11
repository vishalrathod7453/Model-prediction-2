import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import requests

# --- RESILIENT IMPORT ---
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Deployment Pro", page_icon="🚀", layout="wide")

# --- MODERN UI STYLE ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
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

lottie_ani = load_lottieurl("https://lottie.host/825441ec-3c35-4277-9877-33a887413c60/X7U0Yw0rSj.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    if os.path.exists("Model1.pkl"):
        with open("Model1.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# --- UI HEADER ---
col1, col2 = st.columns([1, 2])
with col1:
    if LOTTIE_AVAILABLE and lottie_ani:
        st_lottie(lottie_ani, height=250)
    else:
        st.title("🤖")

with col2:
    st.title("Advanced Predictive Portal")
    st.write("Professional deployment of your KNN Classification Model.")

st.divider()

# --- INPUT SECTION ---
if model:
    # Based on your Model1.pkl feature names
    features = ["Age", "Gender", "Education_Level", "City", "AI_Tool_Used", "Daily_Usage_Hours", "Purpose"]
    
    st.subheader("📝 Input Parameters")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    inputs = {}
    c1, c2 = st.columns(2)
    for i, feat in enumerate(features):
        with c1 if i % 2 == 0 else c2:
            # Note: Ensure these inputs match the numeric encoding used in training
            inputs[feat] = st.number_input(f"Enter {feat}", value=0.0)
    
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("RUN PREDICTION"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)
        
        st.balloons()
        st.success(f"### Predicted Result: {prediction[0]}")
else:
    st.warning("⚠️ Model1.pkl not found. Please upload it to the same folder as app.py.")
