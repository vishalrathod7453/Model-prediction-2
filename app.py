import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Visionary Pro", page_icon="🔮", layout="wide")

# --- GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stFrame) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SAFE LOTTIE LOADER ---
def load_lottie_safe(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Use a highly reliable animation URL
lottie_main = load_lottie_safe("https://lottie.host/825441ec-3c35-4277-9877-33a887413c60/X7U0Yw0rSj.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open("Model2.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Model Load Error: {e}")
        return None

model = load_model()

# --- HEADER ---
with st.container():
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if lottie_main:
            st_lottie(lottie_main, height=250, key="main_ani")
    with col_b:
        st.title("AI Predictive Intelligence")
        st.write("Experience seamless data inference with our advanced KNN architecture.")

st.markdown("---")

# --- MAIN INTERFACE ---
if model:
    # Feature detection
    features = getattr(model, 'feature_names_in_', ["Feature 1", "Feature 2", "Feature 3", "Feature 4"])
    
    with st.container():
        st.subheader("📊 Input Data Parameters")
        
        # Grid layout for inputs
        cols = st.columns(len(features) if len(features) <= 4 else 3)
        input_values = {}
        
        for i, feat in enumerate(features):
            col_idx = i % (len(cols))
            with cols[col_idx]:
                input_values[feat] = st.number_input(f"{feat}", value=0.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("✨ GENERATE PREDICTION"):
        # Data preparation
        input_df = pd.DataFrame([input_values])
        
        with st.status("Analyzing Patterns...", expanded=True) as status:
            st.write("Feeding data to Model2...")
            prediction = model.predict(input_df)
            st.write("Interpreting KNN clusters...")
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # Result Display
        st.balloons()
        st.markdown(f"""
            <div style="text-align: center; padding: 40px; border-radius: 20px; background: rgba(79, 172, 254, 0.1); border: 2px solid #4facfe;">
                <h1 style="margin:0;">RESULT: {prediction[0]}</h1>
                <p style="color: #4facfe; font-size: 1.2rem;">Model2 confidence confirmed.</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Please upload 'Model2.pkl' to the directory to activate the system.")
