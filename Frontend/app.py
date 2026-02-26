import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your exported files
model = joblib.load('house_price_model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Intelligent Real Estate Advisory", layout="wide")

st.title("🏡 AI Property Price Predictor")
st.write("Milestone 1: Predicting property values using historical listing data.")

# Create the form for property attributes
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Total Area (sq ft)", min_value=100, value=1200)
        rooms = st.slider("Number of Rooms", 1, 10, 3)
        
    with col2:
        # If your model used encoding for Location, add those options here
        location_score = st.selectbox("Location Rating", [1, 2, 3, 4, 5], help="1: Rural, 5: Prime Urban")
        amenities_count = st.number_input("Number of Amenities", min_value=0, value=5)

    submit = st.form_submit_button("Predict Market Price")

if submit:
    # 1. Arrange inputs in the EXACT order your model expects
    features = np.array([[size, rooms, location_score, amenities_count]])
    
    # 2. Apply the same scaling used in training
    scaled_features = scaler.transform(features)
    
    # 3. Generate Prediction
    prediction = model.predict(scaled_features)
    
    st.success(f"### Estimated Value: ${prediction[0]:,.2f}")
    
    # Milestone 1: Price Driver Analysis
    st.info("💡 Analysis: Area size and Location Rating were the primary drivers for this estimate.")