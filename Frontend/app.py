import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Apply custom styling and page config
st.set_page_config(
    page_title="Intelligent Real Estate Advisory",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
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

@st.cache_resource
def load_models():
    # Build absolute path relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'Model', 'house_price_model.joblib')
    scaler_path = os.path.join(base_dir, 'Model', 'scaler.joblib')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure you are running this from the correct directory.")
    st.stop()

st.title("🏡 AI Property Price Predictor")
st.markdown("### Predict real estate property values with high accuracy using our advanced Machine Learning model.")
st.write("---")

with st.sidebar:
    st.header("💡 How to use")
    st.write("1. Fill in the property details.")
    st.write("2. Make sure latitude and longitude are accurate.")
    st.write("3. Click Predict Market Price below.")
    st.write("---")

# Create the form for property attributes
with st.form("prediction_form"):
    
    st.subheader("1. General Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("No of Bedrooms", min_value=1, max_value=30, value=3, step=1)
    with col2:
        bathrooms = st.number_input("No of Bathrooms", min_value=0.5, max_value=10.0, value=2.0, step=0.25)
    with col3:
        floors = st.number_input("No of Floors", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
        
    st.write("---")
    st.subheader("2. Area Details (in Sqft)")
    col4, col5, col6 = st.columns(3)
    with col4:
        flat_area = st.number_input("Flat Area", min_value=100.0, value=1500.0, step=10.0)
        living_area_renov = st.number_input("Living Area after Renovation", min_value=100.0, value=1500.0, step=10.0)
    with col5:
        lot_area = st.number_input("Lot Area", min_value=100.0, value=5000.0, step=10.0)
        lot_area_renov = st.number_input("Lot Area after Renovation", min_value=100.0, value=5000.0, step=10.0)
    with col6:
        basement_area = st.number_input("Basement Area", min_value=0, value=0, step=10)
        # We calculate area from basement dynamically or let user input it
        area_from_basement = st.number_input("Area from Basement", min_value=100.0, value=1500.0, step=10.0, help="Typically Flat Area - Basement Area")

    st.write("---")
    st.subheader("3. Location & Build Year")
    col7, col8, col9 = st.columns(3)
    with col7:
        latitude = st.number_input("Latitude", value=47.5112, format="%.4f")
    with col8:
        longitude = st.number_input("Longitude", value=-122.257, format="%.4f")
    with col9:
        age_of_house = st.number_input("Age of House (in Years)", min_value=0, value=30, step=1)
        renovated_year = st.number_input("Renovated Year", min_value=0, value=0, step=1, help="0 if never renovated")
        
    st.write("---")
    st.subheader("4. Condition & Quality")
    col10, col11, col12 = st.columns(3)
    with col10:
        overall_grade = st.slider("Overall Grade", 1, 10, 7, help="1: Poor, 10: Excellent")
    with col11:
        waterfront = st.selectbox("Waterfront View", ["No", "Yes"])
    with col12:
        condition = st.selectbox("Condition of the House", ["Bad", "Fair", "Good", "Excellent", "Okay"])
        
    st.write("---")
    st.subheader("5. Prior History")
    col13, col14 = st.columns(2)
    with col13:
        visited = st.selectbox("No of Times Visited", ["None", "Once", "Twice", "Thrice", "Four"])

    st.write(" \n")
    submit = st.form_submit_button("Predict Market Price", use_container_width=True)

if submit:
    # --- PREPARE INPUT DATA ---
    # Map purely logical inputs
    waterfront_mapped = 1 if waterfront == 'Yes' else 0
    
    # Conditional mappings (Dummy Variables exact match to DF)
    # df1.columns order logic (Condition)
    cond_exc = 1 if condition == 'Excellent' else 0
    cond_fair = 1 if condition == 'Fair' else 0
    cond_good = 1 if condition == 'Good' else 0
    cond_okay = 1 if condition == 'Okay' else 0
    
    # df1.columns order logic (Visited)
    vis_once = 1 if visited == 'Once' else 0
    vis_thrice = 1 if visited == 'Thrice' else 0
    vis_twice = 1 if visited == 'Twice' else 0

    # Ensure array follows exact sequence:
    # 0: No of Bedrooms, 1: No of Bathrooms, 2: Flat Area, 3: Lot Area, 4: No of Floors
    # 5: Waterfront View, 6: Overall Grade, 7: Area from Basement, 8: Basement Area
    # 9: Age, 10: Renovated Year, 11: Latitude, 12: Longitude, 13: Living Area after Renov
    # 14: Lot Area after Renov
    # 15: Cond_Excellent, 16: Cond_Fair, 17: Cond_Good, 18: Cond_Okay
    # 19: Vis_Once, 20: Vis_Thrice, 21: Vis_Twice

    features = np.array([
        bedrooms,
        bathrooms,
        flat_area,
        lot_area,
        floors,
        waterfront_mapped,
        overall_grade,
        area_from_basement,
        basement_area,
        age_of_house,
        renovated_year,
        latitude,
        longitude,
        living_area_renov,
        lot_area_renov,
        cond_exc,
        cond_fair,
        cond_good,
        cond_okay,
        vis_once,
        vis_thrice,
        vis_twice
    ]).reshape(1, -1)
    
    with st.spinner('Analyzing property data and scaling features...'):
        try:
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Predict
            prediction = model.predict(scaled_features)
            
            # Display Result
            st.write("---")
            st.subheader("📊 Analysis Complete")
            st.metric(label="Estimated Property Value", value=f"${prediction[0]:,.2f}")
            
            # Add dynamic feedback
            st.success("Successfully generated market value estimate based on input housing parameters.")
            
        except ValueError as e:
            st.error(f"Shape Mismatch Error: Please verify the model input features. Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")