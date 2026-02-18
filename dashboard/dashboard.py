"""
Dashboard for Wine Detector
Runs locally and displays real-time predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime
import os

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Wine Detector",
    page_icon="🍷",
    layout="wide"
)

# ============================================
# Title and Description
# ============================================
st.title("Wine Detector with BME688")
st.markdown("---")

# ============================================
# Sidebar - Configuration
# ============================================
with st.sidebar:
    st.header("Configuration")
    
    # API URL
    api_url = st.text_input(
        "API URL",
        #value="http://localhost:5000/predict", # URL de localhost
        value="https://anamariascheau.pythonanywhere.com/predict", #URL Cloud
        help="URL of the prediction API"
    )
    
    # Test connection button
    if st.button("Test API Connection"):
        try:
            test_url = api_url.replace("/predict", "/health")
            response = requests.get(test_url, timeout=2)
            if response.status_code == 200:
                st.success("API connected successfully")
            else:
                st.error(f"Connection error: {response.status_code}")
        except Exception as e:
            st.error(f"Cannot connect: {str(e)}")
    
    st.markdown("---")
    
    # Model Information
    st.header("Model Information")
    try:
        info_url = api_url.replace("/predict", "/info")
        response = requests.get(info_url, timeout=2)
        if response.status_code == 200:
            info = response.json()
            st.write(f"**Model type:** {info['model_type']}")
            st.write(f"**Classes:** {', '.join(info['classes'])}")
            st.write(f"**Features:** {', '.join(info['feature_columns'])}")
    except:
        st.write("Model information unavailable")

# ============================================
# Initialize Session State
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'temperature', 'humidity', 
        'gas_resistance', 'iaq', 'prediction', 'confidence'
    ])

if 'running' not in st.session_state:
    st.session_state.running = False

# ============================================
# Main Layout
# ============================================

# Current reading columns
col1, col2, col3, col4 = st.columns(4)

current_prediction = st.empty()
current_confidence = st.empty()

# Control buttons
col_start, col_stop, col_clear = st.columns([1, 1, 2])

with col_start:
    if st.button("Start Monitoring"):
        st.session_state.running = True

with col_stop:
    if st.button("Stop Monitoring"):
        st.session_state.running = False

with col_clear:
    if st.button("Clear History"):
        st.session_state.history = pd.DataFrame(columns=[
            'timestamp', 'temperature', 'humidity', 
            'gas_resistance', 'iaq', 'prediction', 'confidence'
        ])

st.markdown("---")

# ============================================
# Function for Simulation (when no sensor is available)
# ============================================
def generate_test_data():
    """Generate test data for demo purposes"""
    import numpy as np
    
    # Simulate different types
    types = {
        'air': {'temp': 22, 'hum': 45, 'gas': 200000, 'iaq': 25},
        'red_wine': {'temp': 23, 'hum': 65, 'gas': 80000, 'iaq': 120},
        'white_wine': {'temp': 23, 'hum': 70, 'gas': 120000, 'iaq': 90}
    }
    
    # Randomly choose a type
    chosen = np.random.choice(list(types.keys()))
    params = types[chosen]
    
    # Add noise
    return {
        'temperature': params['temp'] + np.random.normal(0, 1),
        'humidity': params['hum'] + np.random.normal(0, 5),
        'gas_resistance': params['gas'] + np.random.normal(0, 10000),
        'iaq': params['iaq'] + np.random.normal(0, 10)
    }

# ============================================
# Main Monitoring Loop
# ============================================
if st.session_state.running:
    st.info("Monitoring active - waiting for data")
    
    # Containers for updates
    status_container = st.empty()
    chart_container = st.empty()
    
    use_simulated = st.sidebar.checkbox("Use simulated data (no sensor)", key="simulated_mode")
    
    while st.session_state.running:
        try:
            if use_simulated:
                # Generate simulated data
                test_data = generate_test_data()
                data_to_send = test_data
            else:
                # Wait for ESP32 data (using simulated for demo)
                data_to_send = generate_test_data()

            
            # Send to API
            response = requests.post(api_url, json=data_to_send, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to history
                new_row = pd.DataFrame([{
                    'timestamp': datetime.now(),
                    'temperature': result['input_data']['temperature'],
                    'humidity': result['input_data']['humidity'],
                    'gas_resistance': result['input_data']['gas_resistance'],
                    'iaq': result['input_data']['iaq'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                }])
                
                st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)
                
                # Keep last 100 records
                if len(st.session_state.history) > 100:
                    st.session_state.history = st.session_state.history.tail(100)
                
                # Update current values
                with col1:
                    st.metric("Temperature", f"{result['input_data']['temperature']:.1f}°C")
                with col2:
                    st.metric("Humidity", f"{result['input_data']['humidity']:.1f}%")
                with col3:
                    st.metric("Gas Resistance", f"{result['input_data']['gas_resistance']:.0f} Ω")
                with col4:
                    st.metric("IAQ", f"{result['input_data']['iaq']:.0f}")
                
                # Update prediction
                with current_prediction.container():
                    st.subheader(f"Current Prediction: {result['prediction']}")
                with current_confidence.container():
                    st.subheader(f"Confidence: {result['confidence']:.1%}")
                
                # Display probabilities
                st.subheader("Class Probabilities")
                prob_cols = st.columns(len(result['probabilities']))
                for idx, (cls, prob) in enumerate(result['probabilities'].items()):
                    with prob_cols[idx]:
                        st.metric(cls, f"{prob:.1%}")
                
                # Update charts
                if len(st.session_state.history) > 1:
                    with chart_container.container():
                        fig = px.line(st.session_state.history, x='timestamp', y=['temperature', 'humidity', 'iaq'],
                                    title="Sensor Readings Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                
                status_container.success(f"Data received at {datetime.now().strftime('%H:%M:%S')}")
            
            else:
                status_container.error(f"API Error: {response.status_code}")
            
            time.sleep(2)  # Wait 2 seconds between readings
            
        except Exception as e:
            status_container.error(f"Error: {str(e)}")
            time.sleep(2)
    
    st.warning("Monitoring stopped")

# ============================================
# Display History Table
# ============================================
if not st.session_state.history.empty:
    st.markdown("---")
    st.subheader("Prediction History")
    
    # Format history for display
    display_history = st.session_state.history.copy()
    display_history['timestamp'] = display_history['timestamp'].dt.strftime('%H:%M:%S')
    display_history['confidence'] = display_history['confidence'].apply(lambda x: f"{x:.1%}")
    display_history['temperature'] = display_history['temperature'].apply(lambda x: f"{x:.1f}°C")
    display_history['humidity'] = display_history['humidity'].apply(lambda x: f"{x:.1f}%")
    display_history['gas_resistance'] = display_history['gas_resistance'].apply(lambda x: f"{x:.0f} Ω")
    display_history['iaq'] = display_history['iaq'].apply(lambda x: f"{x:.0f}")
    
    st.dataframe(display_history, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.subheader("Statistics")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("Total Readings", len(st.session_state.history))
    
    with col_stats2:
        most_common = st.session_state.history['prediction'].mode()[0]
        st.metric("Most Common Prediction", most_common)
    
    with col_stats3:
        avg_confidence = st.session_state.history['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1%}")

else:
    st.info("Start monitoring to see predictions")