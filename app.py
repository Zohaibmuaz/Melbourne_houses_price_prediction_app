import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the pre-trained model and scaler using joblib
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # Assuming you saved the scaler with joblib

# The exact column order expected by the model
expected_columns = ['Rooms', 'Type', 'Method', 'Date', 'Postcode', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea']

# Function to encode categorical features (if required)
def encode_input(input_data):
    label_encoder = LabelEncoder()

    # Example: encoding for 'CouncilArea', 'Type', 'Method', and 'Regionname' (if they're categorical)
    input_data['CouncilArea'] = label_encoder.fit_transform(input_data['CouncilArea'].fillna('Unknown'))
    input_data['Type'] = label_encoder.fit_transform(input_data['Type'].fillna('Unknown'))
    input_data['Method'] = label_encoder.fit_transform(input_data['Method'].fillna('Unknown'))
    input_data['Regionname'] = label_encoder.fit_transform(input_data['Regionname'].fillna('Unknown'))

    return input_data

# Function to convert 'Date' column to a numeric format (e.g., days since a reference date)
def convert_date(input_data):
    reference_date = pd.to_datetime('2000-01-01')  # Define a reference date (e.g., start of the dataset)
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    input_data['Date'] = (input_data['Date'] - reference_date).dt.days  # Convert to days since reference date
    return input_data

# Function for scaling features
def scale_input(input_data):
    # Ensure the input data matches the expected features and order
    input_data = input_data[expected_columns]  # Reorder columns to match model training order
    
    # If any missing columns are there, fill them with appropriate values (e.g., zeros or means)
    input_data = input_data.fillna(0)
    
    # Convert 'Date' column to numeric format
    input_data = convert_date(input_data)

    # Scaling the input data
    return scaler.transform(input_data)

# Streamlit UI for inputs
st.title("House Price Prediction")

# Input fields for features (use the same features as used during model training)
rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, value=2, key="rooms")
property_type = st.selectbox('Property Type', options=['House', 'Unit'], key="property_type")  # Add other types if necessary
method = st.selectbox('Method of Sale', options=['Auction', 'Private Sale'], key="method")  # Replace with actual categories
date = st.date_input('Date', value=pd.to_datetime('2023-01-01'), key="date")
postcode = st.number_input('Postcode', min_value=1000, max_value=9999, value=3000, key="postcode")
region_name = st.selectbox('Region Name', options=['North', 'South', 'East', 'West'], key="region_name")  # Replace with actual categories
property_count = st.number_input('Property Count', min_value=1, max_value=100, value=1, key="property_count")
distance = st.number_input('Distance from City Center (km)', min_value=1, max_value=50, value=20, key="distance")
council_area = st.selectbox('Council Area', options=['Area 1', 'Area 2', 'Area 3'], key="council_area")  # Replace with actual categories

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[rooms, property_type, method, date, postcode, region_name, property_count, distance, council_area]],
                          columns=['Rooms', 'Type', 'Method', 'Date', 'Postcode', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea'])

# Process the input data: Encode categorical features
input_data = encode_input(input_data)

# Scale the input data
scaled_input_data = scale_input(input_data)

# Predict button
if st.button('Predict'):
    # Prediction
    prediction = model.predict(scaled_input_data)
    
    # Show the prediction
    st.write(f'Predicted House Price: ${prediction[0]:,.2f}')
