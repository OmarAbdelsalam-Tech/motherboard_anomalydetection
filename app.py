import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
from PIL import Image
import requests
import base64

st.title("Motherboard Classifier")
st.subheader("Developed by [Your Name]")
st.subheader("Upload an image to determine if it's a motherboard.")

# File uploader
file = st.file_uploader("Upload your motherboard image here:", type=["jpg", "jpeg", "png"])

if file:
    st.image(file, caption='Uploaded Image')

# Button to classify image
button = st.button('Classify Image')

if button:
    # Direct download link for the Google Drive file
    direct_url = 'https://drive.google.com/uc?id=1kPMHyr8LXxY_Al6WHjG5JCYrsbCrHvEw'
    response = requests.get(direct_url)
    with open('my_model.h5', 'wb') as f:
        f.write(response.content)

    # Load the model
    model = load_model('my_model.h5')

    # Preprocess the image
    image = Image.open(file)
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized)
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32) / 255.0

    # Predict the class
    prediction = model.predict(img_tensor)

    # Display result
    result_text = "Motherboard" if prediction[0][0] > 0.5 else "Not a Motherboard"
    st.subheader(f"Prediction: {result_text}")
