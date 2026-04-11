import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Model Deployer", page_icon="🚀", layout="wide")

# --- CUSTOM CSS FOR ANIMATION ---
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%); color: white; }
    .stButton>button {
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white; border: none; border-radius: 10px;
        padding: 10px 24px; transition: 0.4s;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0px 10px 20px rgba(0,114,255,0.4); }
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px; padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_predict = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ghp9m6io.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_pqnfmone.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open("Model2.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading Model2.pkl: {e}")
        return None

model = load_model()

# --- UI LAYOUT ---
st.title("✨ Intelligent Prediction Portal")
st.markdown("### Deploying your `Model2` with style.")

if model:
    # Attempt to extract feature names automatically
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
    else:
        # Fallback if names aren't in model
        features = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"] 
        st.warning("Feature names not found in model. Using placeholders.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st_lottie(lottie_predict, height=250)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Input Parameters")
        
        user_inputs = {}
        # Dynamically create inputs based on features
        for feat in features:
            user_inputs[feat] = st.number_input(f"Enter {feat}", value=0.0)
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- PREDICTION LOGIC ---
    if st.button("🚀 Run Analysis"):
        with st.spinner("Processing..."):
            time.sleep(1) # Visual delay for effect
            
            # Convert inputs to DataFrame
            input_df = pd.DataFrame([user_inputs])
            prediction = model.predict(input_df)
            
            # Display Results
            st.balloons()
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st_lottie(lottie_success, height=150)
            with res_col2:
                st.markdown(f"""
                <div style="background: rgba(0, 255, 0, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #00ff00;">
                    <h2 style="color: #00ff00; margin: 0;">Result: {prediction[0]}</h2>
                    <p style="color: #ccc;">The model has successfully analyzed your inputs.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Model2.pkl not found! Please ensure the file is in the same directory.")
