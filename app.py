import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import cv2
import os
import PIL
from PIL import ImageEnhance
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from streamlit_lottie import st_lottie
import json

st.set_page_config(layout="wide", page_title="GrowPro", page_icon="🌱")

image = ''

image1 = st.container()
info = st.container()
map = st.container()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
logo = load_lottiefile("Animation - 1710065948380.json")
# Sample dataset for demonstration purposes
# Define the coordinates for three locations
locations = {
    'Nidadavole': [(16.913750, 81.656250), (16.909062, 81.658937), (16.897062, 81.663438)],
    'Bhimavaram': [(16.555056207478007, 81.50648383316177), (16.544646454168657, 81.48510664214669), (16.550437720412898, 81.55752927886968)],
    'Palakollu': [(16.510252921174228, 81.75282223695348), (16.518293962948686, 81.70834967429242), (16.51492721068963, 81.74101133722162)],
}
labels = {0: 'Bacterial_leaf_blight', 1: 'Brown_spot', 2: 'Leaf_smut'}
# Create a selectbox for the user to choose a location
loaded_model = keras.models.load_model('rice_leaf_disease_model.h5')
data = pd.read_csv('data.csv',index_col=0)
transposed_data = data.T
data2 = pd.read_csv('info0.csv')
data3 = pd.read_csv('info1.csv',index_col=0)
transposed_data2 = data3.T
predict_button = False
predicted_class = ""  # Initialize predicted_class outside the if block
def preprocess(image):
    # Convert NumPy array to PIL image
    image = PIL.Image.fromarray((image * 255).astype('uint8'))

    # Resize the image to (224, 224)
    image = image.resize((224, 224))
    
    # Normalize pixel values
    image = np.array(image) / 255.0
    
    # Apply image sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    
    # Apply image enhancement
    enhancer = PIL.ImageEnhance.Contrast(PIL.Image.fromarray((image * 255).astype('uint8')))
    image = np.array(enhancer.enhance(1.5)) / 255.0  # adjust the enhancement factor as needed
    
    # Adjust image intensity
    image = image * 1.2  # adjust the intensity factor as needed
    
    return image

def predict_label(image_path):
    # Load the image file
    image = load_img(image_path, target_size=(224, 224))

    # Convert the image to a NumPy array
    image = img_to_array(image)

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Reshape the image to match the input shape of the model
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Predict the label of the image
    prediction = loaded_model.predict(preprocessed_image)
    label = np.argmax(prediction)
    # Return the predicted label
    return labels[label]

st.sidebar.text_input("Enter your Name")
selected_location = st.sidebar.selectbox('Select Your area:', list(locations.keys()))

with st.sidebar:
    input_type = st.selectbox("Pick one", ["Upload Image","Camera Input", ])

    if input_type == "Camera Input":
        image = st.camera_input("Pick a snapshot")
    else:
        image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predict")





with image1:
    st.markdown(
        """
        <style>
            .title {
                text-align: center;
                font-weight: bold;
            }
            .mainheading {
                text-align: center;
                font-family: monospace;
                font-size: 25px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h2 class="title">Grow Pro</h2>', unsafe_allow_html=True)
    st.markdown('<h3 class="mainheading">A Farmer Assistant</h3>',unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,1])
    with c2:
        st_lottie(logo, speed=1, width=400, height=400)
    if input_type == "Upload Image" and image is not None:
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    if predict_button:
        predicted_class = predict_label(image)
        #st.write("Predicted Class:", predicted_class)
        #st.write("Predicted Class: Bacterial leaf blight")

if predict_button:
   
    with info:
        c1, c2 = st.columns([1, 1])
        with c1: 
            st.markdown('<h2 class="title">Information</h2>', unsafe_allow_html=True)
            title = predicted_class
            if predicted_class == "Bacterial_leaf_blight":
                text = "Bacterial leaf blight is a common disease in rice. It is caused by the bacteria Xanthomonas oryzae pv. oryzae. The disease is characterized by water-soaked lesions that later turn brown and necrotic. The lesions are usually found on the leaf tips and margins. The bacteria can also infect the panicle, causing the panicle to rot. The disease is spread by wind, rain, and irrigation water. The bacteria can also be spread by infected seeds. The disease can be controlled by planting resistant varieties, using clean seeds, and applying copper-based bactericides."
            elif predicted_class == "Brown_spot":
                text = "Brown spot is a common disease in rice. It is caused by the fungus Cochliobolus miyabeanus. The disease is characterized by small, brown, water-soaked lesions that later turn necrotic. The lesions are usually found on the leaf tips and margins. The disease is spread by wind, rain, and irrigation water. The fungus can also be spread by infected seeds. The disease can be controlled by planting resistant varieties, using clean seeds, and applying fungicides."
            elif predicted_class == "Leaf_smut":
                text = "Leaf smut is a common disease in rice. It is caused by the fungus Tilletia barclayana. The disease is characterized by small, black, powdery spores that are found on the leaf surface. The spores are spread by wind, rain, and irrigation water. The disease can be controlled by planting resistant varieties, using clean seeds, and applying fungicides."
            
            html_code = f"""
            <div style='
                background-color: #2A2937;
                border-radius: 5px;
                padding: 20px;
                font-family: Arial;
                font-size: 20px;
                border: 2px;
                '>
                <h3>{title}</h3>
                <div style='
                    font-family: monospace;
                    font-size: 18px;
                '>
                {text}</div>
            </div>
            """

            st.markdown(html_code, unsafe_allow_html=True)

        with c2:
            st.markdown('<h2 class="title">Symptoms</h2>', unsafe_allow_html=True)
            st.table(transposed_data[predicted_class])

with map:
    if predict_button:
        # Get the coordinates for the selected location
        selected_coordinates = locations[selected_location]

        # Create a DataFrame with the coordinates
        data = pd.DataFrame({
            'LAT': [coord[0] for coord in selected_coordinates],
            'LON': [coord[1] for coord in selected_coordinates],
        })
        st.markdown('<h2 class="title">Map</h2>', unsafe_allow_html=True)
        st.map(data)
