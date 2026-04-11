import streamlit as st
import joblib
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🚀",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🚀 Smart ML Prediction App")
st.write("### Enter your data and get instant predictions")

# ------------------ LOAD MODEL ------------------
try:
    model = joblib.load("model2.pkl")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ------------------ INPUT SECTION ------------------
st.subheader("📥 Input Features")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)

with col2:
    feature2 = st.number_input("Feature 2", value=0.0)

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict"):
    try:
        input_data = np.array([[feature1, feature2]])
        prediction = model.predict(input_data)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("💡 Built with Streamlit")
