# Gender and Age Prediction Using CNN 


## Overview 📖
This project aims to predict gender and age from facial images using a Convolutional Neural Network (CNN). The UTKFace dataset is used for training and evaluation. The dataset contains images of faces with labels for age, gender, and ethnicity.

## Dataset 📂
- **Name**: UTKFace
- **Description**: A large-scale dataset containing facial images labeled with age, gender, and ethnicity.
- **Source**: [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new/data)
- **Preprocessing**:
  - Resized images to a fixed size (e.g. 128x128)
  - Normalized pixel values
  - Extracted age and gender labels

## Model Architecture 
The CNN model consists of the following layers:
- Convolutional layers with ReLU activation
- BatchNormalization
- MaxPooling layers for downsampling
- Fully connected (dense) layers
- Softmax activation for classification

## Training Details 🎯
- **Loss Function**: 
  - Binary Crossentropy for gender classification (binary classification)
  - Mean Absolute Error (MAE) for age prediction (regression)
- **Optimizer**: Adam
- **Metrics**: Accuracy for gender classification, MAE for age prediction
- **Training**:
  - Split data into training and validation sets (e.g., 80%-20%)
  - Trained for a fixed number of epochs

## Results 📊
- Achieved high accuracy on gender classification (~90%)
- Age prediction had a Mean Absolute Error (MAE) of around 6-8 years

## Files in the Repository 📁
- `Gender-Age-Prediction.ipynb` - Jupyter Notebook containing model training and evaluation
- `real_time.py` - openCv for real time testing
- `app.py` - Streamlit web app for predicting gender and age from uploaded images
- `model.pkl` - Saved trained model
- `requirements.txt` - List of dependencies required to run the project

## How to Run 🏃‍♂️
1. Clone the repository:
   ```bash
   git clone https://github.com/abdelilah2003/Real-time-Age-Gender-Prediction.git
   cd Real-time-Age-Gender-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
 4. Run the Streamlit app:
   ```bash
   python real_time.py
   ```  

## Dependencies 📦
- TensorFlow/Keras
- NumPy
- Matplotlib
- Pandas



---
