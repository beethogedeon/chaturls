import streamlit as st
import requests

st.title("Q/A System for ReventePro")

st.subheader("Ask your question here :")
question = st.text_input("")
button = st.button("Ask")

if button:
    responses = requests.get("https://3181-34-28-199-1.ngrok-free.app/answer?query=" + question)
    responses = responses.json()
    st.success(responses["response"])
