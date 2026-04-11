import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Intelligence Portal", page_icon="🧪", layout="wide")

# --- MODERN UI STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    [data-testid="stVerticalBlock"] > div:has(div.stFrame) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1 {
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #92fe9d);
        color: #0f0c29 !important;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FAIL-SAFE ANIMATION LOADER ---
def load_lottie_safe(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Reliable animation sources
lottie_main = load_lottie_safe("https://lottie.host/825441ec-3c35-4277-9877-33a887413c60/X7U0Yw0rSj.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open("Model2.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model2.pkl not found. Please upload it to the repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_main:
            st_lottie(lottie_main, height=250, key="main_ani")
        else:
            st.title("🤖") # Fallback icon
    with col2:
        st.title("Predictive Analytics Engine")
        st.write("Deploying sophisticated data models with a seamless interface.")

st.divider()

# --- INPUT & PREDICTION ---
if model:
    # Get feature names from model or use defaults
    features = getattr(model, 'feature_names_in_', ["Feature 1", "Feature 2", "Feature 3", "Feature 4"])
    
    st.subheader("📋 Input Parameters")
    
    # Dynamic grid for inputs
    with st.container():
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                input_data[feat] = st.number_input(f"{feat}", value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("✨ GENERATE ANALYSIS"):
        input_df = pd.DataFrame([input_data])
        
        with st.spinner("Model is calculating..."):
            prediction = model.predict(input_df)
            
            st.balloons()
            st.markdown(f"""
                <div style="background: rgba(146, 254, 157, 0.1); padding: 30px; border-radius: 15px; border: 1px solid #92fe9d; text-align: center;">
                    <h2 style="color: #92fe9d; margin: 0;">Predicted Output: {prediction[0]}</h2>
                    <p style="color: #ccc;">Analysis completed successfully using Model2.</p>
                </div>
            """, unsafe_allow_html=True)
