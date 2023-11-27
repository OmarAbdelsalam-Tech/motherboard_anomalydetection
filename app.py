import streamlit as st
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import requests

# Function to download the model file from Google Drive
def download_model(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Function to preprocess the image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI setup
st.title("Motherboard Classification")
st.write("Upload an image to classify if it's a motherboard or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        with st.spinner('Loading model and classifying...'):
            model_path = 'my_model.h5'
            model_url = 'https://drive.google.com/uc?id=1BLWDHXoA_mwbVKiMIAGcUapdCo47UwNU'  # Your Google Drive file URL
            if not os.path.isfile(model_path):
                download_model(model_url, model_path)

            try:
                model = load_model(model_path)
                image_data = preprocess_image(uploaded_file)
                prediction = model.predict(image_data)
                class_label = "Motherboard" if prediction[0][0] > 0.5 else "Not a Motherboard"
                st.success(f"Prediction: {class_label}")
            except Exception as e:
                st.error(f"Error loading model: {e}")
