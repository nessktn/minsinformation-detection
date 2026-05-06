import streamlit as st
from model import predict

st.title("PNG Misinformation Detector")

user_input = st.text_area("Enter a social media post:")

if st.button("Analyze"):
    label, confidence = predict(user_input)

    if label == 1:
        st.error(f"⚠️ Likely Misinformation ({confidence:.2f})")
    else:
        st.success(f"✅ Likely Credible ({confidence:.2f})")