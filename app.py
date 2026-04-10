import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="AI Impact Insights", page_icon="🤖", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 3em;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- ANIMATION LOADER ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_m6cu96ze.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Make sure 'Model (3).pkl' is uploaded to your GitHub main folder
    with open('Model (3).pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'Model (3).pkl' not found. Please upload it to your GitHub repository.")
    st.stop()

# --- FRONTEND ---
st.title("🎓 Student AI Usage & Impact Predictor")
st.write("Predict student outcomes based on their interaction with AI tools.")

col1, col2 = st.columns([1, 1.2])

with col1:
    st_lottie(lottie_ai, height=400, key="coding")

with col2:
    st.subheader("📊 Input Student Parameters")
    with st.container():
        # Feature inputs organized to match the 8 expected features
        age = st.number_input("Age", 10, 80, 20)
        
        # Note: KNN models require numerical inputs. 
        # These select boxes map common categories to numbers (0, 1, 2...)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        edu = st.selectbox("Education Level", [0, 1, 2], format_func=lambda x: ["School", "UG", "PG"][x])
        city = st.number_input("City Code (e.g., 0-10)", 0, 10, 1)
        
        ai_tool = st.selectbox("AI Tool Used", [0, 1, 2], format_func=lambda x: ["ChatGPT", "Gemini", "Other"][x])
        hours = st.slider("Daily Usage Hours", 0, 12, 2)
        purpose = st.selectbox("Purpose", [0, 1], format_func=lambda x: "Academic" if x==0 else "Personal")
        impact = st.selectbox("Current Impact on Grades", [0, 1], format_func=lambda x: "Neutral" if x==0 else "Positive")

    if st.button("✨ Run AI Prediction"):
        # Construct exactly 8 features in the correct order 
        input_features = np.array([[age, gender, edu, city, ai_tool, hours, purpose, impact]])
        
        try:
            prediction = model.predict(input_features)
            st.balloons()
            st.success(f"### Predicted Result: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.divider()
st.caption("AI Model Deployment | 2026 Professional Portfolio")
