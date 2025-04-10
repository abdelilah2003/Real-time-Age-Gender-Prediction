import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import matplotlib.pyplot as plt

# Load model
def load_model():
    # Replace with your actual path to the saved pickle file
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocessing function as provided
def get_image_features(image):
    # Load the image in grayscale
    img = load_img(image, color_mode='grayscale')
    img = img.resize((128, 128), Image.Resampling.LANCZOS)  # Resize the image
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)  # Reshape for model input
    img = img / 255.0  # Normalize image
    return img

# Streamlit interface
def main():
    st.title("UTKFace Age and Gender Prediction")
    st.write("Upload an image to predict the age and gender.")

    # Allow user to upload image
    img_to_test = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if img_to_test is not None:

        # Preprocess image and predict results
        processed_image = get_image_features(img_to_test)
        
        # Load the pre-trained model
        model = load_model()

        # Make prediction
        pred = model.predict(processed_image)

        # Map gender to appropriate labels
        gender_mapping = {0: 'Male', 1: 'Female'}
        gender = gender_mapping[round(pred[0][0][0])]  # Male or Female
        age = round(pred[1][0][0])  # Predicted Age

        # Show the result plot with the image
        plt.title(f'Predicted Age: {age} Predicted Gender: {gender}')
        plt.axis('off')
        plt.imshow(np.array(load_img(img_to_test)))  # show the uploaded image
        st.pyplot(plt)

if __name__ == "__main__":
    main()