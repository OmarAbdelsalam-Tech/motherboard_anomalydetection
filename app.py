
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import os
from PIL import Image
global file
import requests
import base64
import streamlit as st
import base64


st.title("This Website uses Deep Learning to Binary Classify Pizza if it's cooked or not yet.")
st.subheader("Contributions: Omar Abdelsalam")
st.subheader("Please provide an image of your pizza!")
    
  
import base64

 
    
st.subheader("Upload your pizza Image here:")

file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])
    
if file:
    st.image(file,caption=' ')
        
button = st.button('Click me and find out if your pizza is BUSSIN or DISGUSTING')
        
if button:
         import requests

         url = 'https://drive.google.com/file/d/1w3S7MeK7GkicfkULCCrD0gMfFmN-8DM0'
         response = requests.get(url)

         with open('my_model.h5', 'wb') as f:
                f.write(response.content)
            
         from keras.models import Sequential, load_model
         new_model = load_model('my_model.h5')
        
         #optimizer = Adam(learning_rate=0.001, decay=1e-6)
         #new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        

         image = Image.open(file)
                
         

        # Resize the image to 256x256 pixels
         img_resized = image.resize((256, 256))

        # Convert the PIL image to a numpy array
         img_array = np.array(img_resized)

        # Convert the numpy array to a TensorFlow tensor
         img_tensor = tf.convert_to_tensor(img_array)

        # Add an extra dimension to the tensor to represent the batch size (1)
         img_tensor = tf.expand_dims(img_tensor, axis=0)

        # Normalize the image tensor
        
         img_tensor = tf.cast(img_tensor, tf.float32)
         img_tensor = img_tensor / 255.0

     
    
         yhat = new_model.predict(img_tensor)
         
         import time

         progress_text = "Watch as our AI technology leaps into action and paves the way for a smarter future!"

         my_bar = st.progress(0, text=progress_text)

         for percent_complete in range(100):
                time.sleep(0.02)
                my_bar.progress(percent_complete + 1, text=progress_text)
        
                
         if 0 <= yhat <= 0.20: 
            st.subheader(f'THIS IS COOKED PIZZA!') 
            st.balloons()   
            
         elif 0.21 <= yhat <= 0.40:
            st.subheader('its cooked but not crispy')
            st.balloons()
         elif 0.41 <= yhat <= 0.60:
            st.subheader('Needs more time')
            
         elif 0.61 <= yhat <= 1.0:
        
            st.subheader(f'ITS RAWW - Ramsey')
           
pass



                                    
