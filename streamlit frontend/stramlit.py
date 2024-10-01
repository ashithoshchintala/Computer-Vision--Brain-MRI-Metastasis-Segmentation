import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Send the image to FastAPI for prediction
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    
    if response.status_code == 200:
        prediction = response.json()
        st.write("Segmentation result:", prediction['segmentation_result'])
