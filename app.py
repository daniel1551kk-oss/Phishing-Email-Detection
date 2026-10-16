
import streamlit as st
import re
from phishing_detector import model, extract_features

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Phishing Email Detection")
st.write("Analyze email text using a machine learning model.")
st.info("This is a demonstration model. Predictions are not a guarantee of email safety.")

email = st.text_area(
    "Paste email content here",
    height=180,
    placeholder="Enter the email subject and message..."
)

if st.button("🔍 Detect Phishing", use_container_width=True):
    if not email.strip():
        st.warning("Please enter an email first.")
    else:
        processed = extract_features(email)
        prediction = model.predict([processed])[0]

        probabilities = model.predict_proba([processed])[0]
        classes = list(model.named_steps["classifier"].classes_)
        confidence = probabilities[classes.index(prediction)] * 100

        if prediction == "Phishing":
            st.error("⚠️ Prediction: Phishing")
            st.write("Avoid clicking suspicious links or sharing personal information.")
        else:
            st.success("✅ Prediction: Safe")
            st.write("No phishing pattern was detected, but verify the sender and links.")

        st.metric("Model confidence", f"{confidence:.2f}%")
        st.caption(
            "Confidence reflects the model's estimate, not the probability "
            "that this email is genuinely malicious or safe."
        )
