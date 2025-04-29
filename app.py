import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Expected columns
expected_columns = ['Rooms', 'Type', 'Method', 'Date', 'Postcode', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea']

# Function to encode categorical features
def encode_input(input_data):
    label_encoder = LabelEncoder()
    input_data['CouncilArea'] = label_encoder.fit_transform(input_data['CouncilArea'].fillna('Unknown'))
    input_data['Type'] = label_encoder.fit_transform(input_data['Type'].fillna('Unknown'))
    input_data['Method'] = label_encoder.fit_transform(input_data['Method'].fillna('Unknown'))
    input_data['Regionname'] = label_encoder.fit_transform(input_data['Regionname'].fillna('Unknown'))
    return input_data

# Function to convert date
def convert_date(input_data):
    reference_date = pd.to_datetime('2000-01-01')
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    input_data['Date'] = (input_data['Date'] - reference_date).dt.days
    return input_data

# Function for scaling
def scale_input(input_data):
    input_data = input_data[expected_columns]
    input_data = input_data.fillna(0)
    input_data = convert_date(input_data)
    return scaler.transform(input_data)

# --- Streamlit App UI ---

# Inject CSS for custom styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px 0px #d3d3d3;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title {
        font-size: 48px;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #333333;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="title">🏡 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown("<h3 style='color: yellow;'>Predict house prices with ease using our smart ML model</h3>", unsafe_allow_html=True)


# Input section inside a container
with st.container():
    st.header("Enter House Details:")
    
    col1, col2 = st.columns(2)

    with col1:
        rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, value=2, key="rooms")
        property_type = st.selectbox('Property Type', options=['House', 'Unit'], key="property_type")
        method = st.selectbox('Method of Sale', options=['Auction', 'Private Sale'], key="method")
        council_area = st.selectbox('Council Area', options=['Area 1', 'Area 2', 'Area 3'], key="council_area")
        property_count = st.number_input('Property Count', min_value=1, max_value=100, value=1, key="property_count")
    
    with col2:
        date = st.date_input('Date', value=pd.to_datetime('2023-01-01'), key="date")
        postcode = st.number_input('Postcode', min_value=1000, max_value=9999, value=3000, key="postcode")
        region_name = st.selectbox('Region Name', options=['North', 'South', 'East', 'West'], key="region_name")
        distance = st.number_input('Distance from City Center (km)', min_value=1, max_value=50, value=20, key="distance")

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[rooms, property_type, method, date, postcode, region_name, property_count, distance, council_area]],
                          columns=['Rooms', 'Type', 'Method', 'Date', 'Postcode', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea'])

# Process input
input_data = encode_input(input_data)
scaled_input_data = scale_input(input_data)

# Prediction
st.markdown("<br>", unsafe_allow_html=True)
if st.button('Predict'):
    # Simulate analysis time with a progress bar
    progress_text = "Analyzing the data. Please wait..."
    progress_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)  # 2 seconds total (100 x 0.02)
        progress_bar.progress(percent_complete + 1, text=progress_text)

    # Once done, clear progress bar and show prediction
    prediction = model.predict(scaled_input_data)
    st.success(f'Predicted House Price: ${prediction[0]:,.2f}')

# Footer
st.markdown("<hr style='margin-top:40px;'>", unsafe_allow_html=True)
st.markdown("<center>Made with ❤️ by Zohaib_Muaz</center>", unsafe_allow_html=True)
