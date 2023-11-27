import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import io
import requests
import os

# Function to download the model file from Google Drive
def download_model(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

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
    # Show uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        # Display a progress bar
        with st.spinner('Loading model and classifying...'):
            # Download the model
            model_path = 'my_model.h5'
            model_file_id = '1BLWDHXoA_mwbVKiMIAGcUapdCo47UwNU'  # Your Google Drive file ID
            if not os.path.isfile(model_path):
                download_model(model_file_id, model_path)

            # Load the model
            model = load_model(model_path)

            # Preprocess the image and make a prediction
            image_data = preprocess_image(uploaded_file)
            prediction = model.predict(image_data)
            class_label = "Motherboard" if prediction[0][0] > 0.5 else "Not a Motherboard"
            st.success(f"Prediction: {class_label}")
