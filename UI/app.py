import streamlit as st
import requests
from PIL import Image

API_URL = "https://celeb-image-classification.onrender.com/classify_image"

st.title("Sports Celebrity Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        files = {"image_data": uploaded_file.getvalue()}  # Send raw bytes as file

        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            results = response.json()
            for res in results:
                st.write(f"Class: {res['class']}")
                st.write("Probabilities:")
                for k, v in res['class_dictionary'].items():
                    st.write(f"{k}: {v}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
