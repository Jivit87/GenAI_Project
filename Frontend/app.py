import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configure the Streamlit page layout and metadata
st.set_page_config(
    page_title="Intelligent Real Estate Advisory",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling for the application interface
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

import os

# Function to load the pre-trained machine learning model and feature scaler
@st.cache_resource
def load_models():
    """Loads the property price prediction model and the associated scaler."""
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file_path = os.path.join(base_directory, 'Model', 'house_price_model.joblib')
    scaler_file_path = os.path.join(base_directory, 'Model', 'scaler.joblib')
    
    price_prediction_model = joblib.load(model_file_path)
    feature_scaler = joblib.load(scaler_file_path)
    return price_prediction_model, feature_scaler

# Attempt to load models and handle potential errors gracefully
try:
    price_prediction_model, feature_scaler = load_models()
except Exception as loading_error:
    st.error(f"Error loading models: {loading_error}. Please ensure you are running this from the correct directory.")
    st.stop()

# Set up the main application header and description
st.title("🏡 AI Property Price Predictor")
st.markdown("### Predict real estate property values with high accuracy using our advanced Machine Learning model.")
st.write("---")

# Render instructions in the sidebar
with st.sidebar:
    st.header("💡 How to use")
    st.write("1. Fill in the property details.")
    st.write("2. Make sure latitude and longitude are accurate.")
    st.write("3. Click Predict Market Price below.")
    st.write("---")

# Create a form to collect user inputs for the property features
with st.form("property_prediction_form"):
    
    # Section 1: Basic room and floor information
    st.subheader("1. General Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_bedrooms = st.number_input("No of Bedrooms", min_value=1, max_value=30, value=3, step=1)
    with col2:
        num_bathrooms = st.number_input("No of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    with col3:
        num_floors = st.number_input("No of Floors", min_value=1, max_value=5, value=1, step=1)
        
    st.write("---")
    
    # Section 2: Area measurements for different parts of the property
    st.subheader("2. Area Details (in Sqft)")
    col4, col5, col6 = st.columns(3)
    with col4:
        total_flat_area = st.number_input("Flat Area", min_value=100.0, value=1500.0, step=10.0)
        living_area_after_renovation = st.number_input("Living Area after Renovation", min_value=100.0, value=1500.0, step=10.0)
    with col5:
        total_lot_area = st.number_input("Lot Area", min_value=100.0, value=5000.0, step=10.0)
        lot_area_after_renovation = st.number_input("Lot Area after Renovation", min_value=100.0, value=5000.0, step=10.0)
    with col6:
        basement_area = st.number_input("Basement Area", min_value=0, value=0, step=10)
        area_excluding_basement = st.number_input("Area from Basement", min_value=100.0, value=1500.0, step=10.0, help="Typically Flat Area - Basement Area")

    st.write("---")
    
    # Section 3: Geographical and historical age information
    st.subheader("3. Location & Build Year")
    col7, col8, col9 = st.columns(3)
    with col7:
        property_latitude = st.number_input("Latitude", value=47.5112, format="%.4f")
    with col8:
        property_longitude = st.number_input("Longitude", value=-122.257, format="%.4f")
    with col9:
        property_age_years = st.number_input("Age of House (in Years)", min_value=0, value=30, step=1)
        year_renovated = st.number_input("Renovated Year", min_value=0, value=0, step=1, help="0 if never renovated")
        
    st.write("---")
    
    # Section 4: Qualitative assessments of the property
    st.subheader("4. Condition & Quality")
    col10, col11, col12 = st.columns(3)
    with col10:
        property_overall_grade = st.slider("Overall Grade", 1, 10, 7, help="1: Poor, 10: Excellent")
    with col11:
        has_waterfront_view = st.selectbox("Waterfront View", ["No", "Yes"])
    with col12:
        property_condition = st.selectbox("Condition of the House", ["Bad", "Fair", "Good", "Excellent", "Okay"])
        
    st.write("---")
    
    # Section 5: Viewing history
    st.subheader("5. Prior History")
    col13, col14 = st.columns(2)
    with col13:
        times_visited = st.selectbox("No of Times Visited", ["None", "Once", "Twice", "Thrice", "Four"])

    st.write(" \n")
    
    # Submit button for the prediction form
    predict_button_clicked = st.form_submit_button("Predict Market Price", use_container_width=True)

# Process the inputs and generate a prediction when the form is submitted
if predict_button_clicked:
    # Convert categorical waterfront view to numerical binary feature
    is_waterfront_mapped = 1 if has_waterfront_view == 'Yes' else 0
    
    # One-hot encode the property condition
    is_condition_excellent = 1 if property_condition == 'Excellent' else 0
    is_condition_fair = 1 if property_condition == 'Fair' else 0
    is_condition_good = 1 if property_condition == 'Good' else 0
    is_condition_okay = 1 if property_condition == 'Okay' else 0
    
    # One-hot encode the number of times visited
    visited_once = 1 if times_visited == 'Once' else 0
    visited_twice = 1 if times_visited == 'Twice' else 0
    visited_thrice = 1 if times_visited == 'Thrice' else 0

    # Assemble all processed features into a single NumPy array for the model
    # Note: The order must match the feature selection order used during model training
    input_features_array = np.array([
        num_bedrooms,
        num_bathrooms,
        total_flat_area,
        total_lot_area,
        num_floors,
        is_waterfront_mapped,
        property_overall_grade,
        area_excluding_basement,
        basement_area,
        property_age_years,
        year_renovated,
        property_latitude,
        property_longitude,
        living_area_after_renovation,
        lot_area_after_renovation,
        is_condition_excellent,
        is_condition_fair,
        is_condition_good,
        is_condition_okay,
        visited_once,
        visited_thrice,
        visited_twice
    ]).reshape(1, -1)
    
    # Display a loading spinner while processing the prediction
    with st.spinner('Analyzing property data and scaling features...'):
        try:
            # Apply the pre-fitted scaler to normalize user inputs
            scaled_input_features = feature_scaler.transform(input_features_array)
            
            # Predict the property value using the trained model
            predicted_price = price_prediction_model.predict(scaled_input_features)
            
            # Display the result in a highlighted metric component
            st.write("---")
            st.subheader("📊 Analysis Complete")
            st.metric(label="Estimated Property Value", value=f"${predicted_price[0]:,.2f}")
            
            st.success("Successfully generated market value estimate based on input housing parameters.")
            
        except ValueError as value_error:
            # Handle cases where the input feature shape or types are incorrect
            st.error(f"Shape Mismatch Error: Please verify the model input features. Details: {value_error}")
        except Exception as unexpected_error:
            # Catch-all for any other unanticipated errors
            st.error(f"An unexpected error occurred: {unexpected_error}")