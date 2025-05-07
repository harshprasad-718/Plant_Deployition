import streamlit as st
import tensorflow as tf
import numpy as np
import json
from gtts import gTTS
import os
from playsound import playsound
import uuid
import cv2
import requests

with open("disease_info.json", "r") as file:
    disease_info = json.load(file)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1tTyJwTURZhgjTxvfiBQhH2s5paGq45vN"
MODEL_PATH = "trained_model.keras"

def download():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model....")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        

# Tensorflow Model Prediction
def model_prediction(test_image):
    download()
    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction) #return index of max element

#Estimate Severity
def estimate_severity(uploaded_image):
    uploaded_image.seek(0)  # Reset file pointer
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (256, 256))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask for brown/dark spots (typical disease color)
    lower = np.array([5, 50, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    disease_pixels = cv2.countNonZero(mask)
    total_pixels = 256 * 256

    ratio = disease_pixels / total_pixels

    if ratio < 0.05:
        return "Mild"
    elif ratio < 0.20:
        return "Moderate"
    else:
        return "Severe"

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=='Home'):
    st.header('PLANT DISEASE RECOGNITION SYSTEM')
    image_path = 'home_page.jpg'
    st.image(image_path,use_column_width=True)
    st.markdown(""" 
                 Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
                
    """)

#About Page
elif(app_mode=='About'):
    st.header('📊 About the Project')
    
    st.markdown("""
    ### 📂 Dataset Overview
    
    It contains approximately **87,000 RGB images** of both **healthy** and **diseased crop leaves**,  
    categorized into **38 different classes**.

    The dataset is split into:
    - **Training set:** 70,295 images (≈ 80%)
    - **Validation set:** 17,572 images (≈ 20%)
    - **Test set:** 33 images (added later for prediction)

    ### 📦 Dataset Structure

    - 📁 `train/` – 70,295 images  
    - 📁 `validation/` – 17,572 images  
    - 📁 `test/` – 33 images

    Each folder maintains the **directory structure of the original dataset**,  
    which helps in seamless model training and evaluation.

    ---
    """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        #st.snow()
        with st.spinner("Please Wait..."):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            #Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
        
        upredicted_class = class_name[result_index]
        predicted_class = upredicted_class.replace('_',' ')
        st.success("Model is Predicting it's a {}".format(predicted_class))
        def speak_result(prediction_text):
            tts = gTTS(text=f"The detected disease is {prediction_text}", lang='en')
            filename = f"temp_audio_{uuid.uuid4()}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
            
        if upredicted_class in disease_info:
            info = disease_info[upredicted_class]
            st.markdown(f"### 🌱 Disease Info for **{predicted_class}**")
            st.markdown(f"**Cause:** {info['Cause']}")
            st.markdown(f"**Prevention:** {info['Prevention']}")
            st.markdown(f"**Remedies:** {info['Remedies']}")
        else:
            st.warning("No additional information available for this disease.")
        speak_result(predicted_class)   

        #Severity Prediction
        test_image.seek(0) # Reset file pointer before reuse
        severity = estimate_severity(test_image)
        st.markdown(f"### 🩺 Estimated Severity: **{severity}**")
        speak_result(severity)