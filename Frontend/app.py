import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from agent.graph import build_graph

# Load environment variables (API Keys)
load_dotenv()

# Configure the Streamlit page layout and metadata
st.set_page_config(
    page_title="Intelligent Real Estate Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styling for a more professional, real-estate focused look
st.markdown("""
<style>
    .reportview-container .main .block-container{ padding-top: 2rem; }
    h1 {
        background: -webkit-linear-gradient(45deg, #2c3e50, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #2980b9; }
</style>
""", unsafe_allow_html=True)

# Cache models to save memory and time
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'Model', 'house_price_model.joblib')
    scaler_path = os.path.join(base_dir, 'Model', 'scaler.joblib')
    return joblib.load(model_path), joblib.load(scaler_path)

try:
    price_model, feature_scaler = load_models()
    agent_app = build_graph()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

st.title("🏡 Intelligent Real Estate Advisor")
st.markdown("### Milestone 2: AI-Powered Investment Analysis & Property Valuation")

# Sidebar for User Preferences and Mode Selection
with st.sidebar:
    st.header("🎯 Advisor Settings")
    app_mode = st.radio("Choose Mode", ["Quick Price Prediction", "Full AI Advisory Report"])
    
    st.write("---")
    st.subheader("👤 Your Profile")
    user_goal = st.selectbox(
        "Primary Goal", 
        ["Personal Residence", "Long-term Rental", "Flipping / Short-term Profit", "Wealth Preservation"]
    )
    user_budget = st.number_input("Target Budget ($)", min_value=0, value=600000, step=50000)
    
    st.write("---")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ GROQ_API_KEY missing in .env. AI Agent reasoning will be disabled.")

# Main Form for Property Details
with st.form("property_details_form"):
    st.subheader("🏠 Property Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_bedrooms = st.number_input("No of Bedrooms", 1, 15, 3)
        num_bathrooms = st.number_input("No of Bathrooms", 0.5, 10.0, 2.0, 0.25)
        asking_price = st.number_input("Current Asking Price ($)", min_value=1, value=500000, help="What is the seller asking for?")
        
    with col2:
        flat_area = st.number_input("Flat Area (sq ft)", 100, 15000, 1500)
        lot_area = st.number_input("Lot Area (sq ft)", 100, 500000, 5000)
        num_floors = st.number_input("No of Floors", 1.0, 4.0, 1.0, 0.5)

    with col3:
        latitude = st.number_input("Latitude", value=47.5112, format="%.4f")
        longitude = st.number_input("Longitude", value=-122.257, format="%.4f")
        property_condition = st.selectbox("Condition", ["Bad", "Fair", "Good", "Excellent"])

    st.write("---")
    submit_btn = st.form_submit_button("Run Analysis", use_container_width=True)

if submit_btn:
    # Prepare features for the model
    # (Mapping logic remains same as per training requirements)
    is_waterfront = 0
    is_cond_exc = 1 if property_condition == "Excellent" else 0
    is_cond_fair = 1 if property_condition == "Fair" else 0
    is_cond_good = 1 if property_condition == "Good" else 0
    is_cond_okay = 1 if property_condition == "Okay" else 0
    
    # Simple defaults for missing features
    features = {
        "no_of_bedrooms": num_bedrooms,
        "no_of_bathrooms": num_bathrooms,
        "total_flat_area": flat_area,
        "total_lot_area": lot_area,
        "no_of_floors": num_floors,
        "waterfront_view": 0,
        "overall_grade": 7,
        "area_excluding_basement": flat_area,
        "basement_area": 0,
        "age_of_house": 20,
        "renovated_year": 0,
        "latitude": latitude,
        "longitude": longitude,
        "living_area_after_renovation": flat_area,
        "lot_area_after_renovation": lot_area,
        "condition_excellent": is_cond_exc,
        "condition_fair": is_cond_fair,
        "condition_good": is_cond_good,
        "condition_okay": is_cond_okay,
        "visited_once": 0,
        "visited_thrice": 0,
        "visited_twice": 0,
        "asking_price": asking_price
    }

    if app_mode == "Quick Price Prediction":
        with st.spinner("Calculating Baseline Market Value..."):
            # Direct model call for speed
            from agent.tools import predict_property_price
            res = predict_property_price.invoke({"features": features})
            st.success("Analysis Complete")
            st.metric("Estimated Market Value", f"${res['predicted_price']:,.2f}")
            st.info(f"Price Range: ${res['price_range']['low']:,.2f} - ${res['price_range']['high']:,.2f}")
    
    else:
        # FULL AGENTIC WORKFLOW
        with st.spinner("🤖 Agent is reasoning, searching RAG docs, and evaluating market trends..."):
            initial_state = {
                "property_features": features,
                "user_preferences": {"goal": user_goal, "budget": user_budget}
            }
            final_state = agent_app.invoke(initial_state)
            
            report = final_state["advisory_report"]
            
            st.write("---")
            tab1, tab2, tab3 = st.tabs(["📝 Summary & Advice", "🏘️ Comparables", "⚖️ Legal Disclaimer"])
            
            with tab1:
                st.markdown(report["summary"])
                st.write("---")
                st.subheader("💡 Expert Analysis")
                st.markdown(report["analysis"])
                
            with tab2:
                st.subheader("Real Historical Comparables (Nearest Matches)")
                st.markdown(report["comparables"])
                
            with tab3:
                st.info(report["disclaimer"])
