import streamlit as st
import requests
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import io

# Function to download a model file from Google Drive
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
            if chunk:
                f.write(chunk)

# Download the .h5 model file
model_path = 'inception_feature_extractor.h5'
model_file_id = '1_PrcaInABNl_7clMqae8KTxQFX5BGf-X'

if not os.path.isfile(model_path):
    st.write("Downloading TensorFlow model...")
    download_model(model_file_id, model_path)
    st.write("Download complete.")

# Download the .pkl model file
lof_model_path = 'lof_model.pkl'
lof_model_file_id = '1YB1tJelTCqFtj2xlz5qd1n47Go2txBNG'

if not os.path.isfile(lof_model_path):
    st.write("Downloading LOF model...")
    download_model(lof_model_file_id, lof_model_path)
    st.write("Download complete.")

# Function to preprocess and extract features from an image
def extract_features(img_data, model):
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to detect anomalies
def detect_anomaly(new_image_data, model, clf):
    new_image_features = extract_features(new_image_data, model)
    anomaly_score = clf.decision_function([new_image_features])[0]
    is_anomaly = anomaly_score < 0
    return "Anomaly Detected (Not a Motherboard)" if is_anomaly else "Motherboard Detected"

# Load the models
feature_model = load_model(model_path) if os.path.isfile(model_path) else None
clf = joblib.load(lof_model_path) if os.path.isfile(lof_model_path) else None

# Streamlit interface
st.title("Motherboard Anomaly Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Detect anomaly
    if st.button('Detect Anomaly'):
        result = detect_anomaly(uploaded_file.read(), feature_model, clf)
        st.write(result)
