import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import os
import time

st.title("Motherboard Classification App")
st.subheader("Contributions: [Your Name]")
st.subheader("Please upload an image of the motherboard!")

# Function to download the model file
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

# Streamlit UI for uploading an image
st.subheader("Upload your motherboard image here:")
file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])
if file:
    st.image(file, caption='Uploaded Image')

# Button to classify the image
button = st.button('Classify Image')
if button:
    # Show progress text and progress bar
    progress_text = "Analyzing the image with our deep learning model..."
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1)

    # Download and load the model
    model_url = 'https://drive.google.com/file/d/1Ks_cTuDqNBqNya614tTQgwl87Mqj6gZX/view?usp=sharing'
    model_path = 'my_model.keras'
    if not os.path.isfile(model_path):
        download_model(model_url, model_path)
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('my_model.keras', compile=False)
        image_data = preprocess_image(file)
        prediction = model.predict(image_data)
        class_label = "Motherboard" if prediction[0][0] > 0.5 else "Not a Motherboard"
        st.subheader(f'Prediction: {class_label}')
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
