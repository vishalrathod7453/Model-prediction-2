import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests

# Try to import lottie, fallback if it fails
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# --- CONFIG & STYLE ---
st.set_page_config(page_title="AI Model Portal", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: white;
    }
    .stNumberInput, .stSelectbox {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(0, 200, 255, 0.1);
        border: 1px solid #00c8ff;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_url = "https://lottie.host/825441ec-3c35-4277-9877-33a887413c60/X7U0Yw0rSj.json"
lottie_json = load_lottieurl(lottie_url)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("Model1.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- UI HEADER ---
col1, col2 = st.columns([1, 2])
with col1:
    if LOTTIE_AVAILABLE and lottie_json:
        st_lottie(lottie_json, height=250)
    else:
        st.title("🤖")

with col2:
    st.title("AI Performance Predictor")
    st.write("Professional deployment of your KNN Classification Model.")

st.divider()

# --- INPUT FORM ---
features = ["Age", "Gender", "Education", "City", "AI Tool", "Usage Hours", "Purpose"]
input_data = {}

with st.container():
    st.subheader("Input Parameters")
    c1, c2 = st.columns(2)
    
    for i, feat in enumerate(features):
        with c1 if i < 4 else c2:
            input_data[feat] = st.number_input(f"Enter {feat}", value=0.0)

# --- PREDICTION ---
if st.button("🚀 Run Prediction"):
    # Convert inputs to the format model expects
    input_df = pd.DataFrame([input_data.values()], columns=features)
    
    prediction = model.predict(input_df)
    
    st.balloons()
    st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: #00c8ff;">Classification Result</h2>
            <h1>{prediction[0]}</h1>
        </div>
    """, unsafe_allow_html=True)
