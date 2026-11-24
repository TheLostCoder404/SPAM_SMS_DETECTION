from pathlib import Path
import json
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
INFO_PATH = ROOT / "models" / "best_model_info.json"

st.set_page_config(page_title="Spam SMS Detection", page_icon="📩", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_info():
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))

st.title("📩 Spam SMS Detection")
st.write("Enter an SMS message below. The machine learning model will classify it as Spam or Ham (Legitimate).")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` first.")
    st.stop()

model = load_model()
info = load_info()
st.caption(f"Selected model: {info.get('best_model', 'Saved best model')}")

message = st.text_area("Enter your SMS message", height=150, placeholder="Example: Congratulations! You have won a free prize...")
if st.button("Predict Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter an SMS message first.")
    else:
        prediction = model.predict([message.strip()])[0]
        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write("This message has been classified as spam. Avoid clicking suspicious links or sharing personal information.")
        else:
            st.success("✅ Prediction: HAM (LEGITIMATE)")
            st.write("This message has been classified as legitimate by the model.")
        st.caption("Predictions are model estimates and may not always be correct.")
st.divider()
st.caption("Built with Python, TF-IDF and scikit-learn.")
