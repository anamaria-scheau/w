"""
REST API for Wine Detector - PYTHONANYWHERE VERSION
This API loads the trained model and makes predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
import sys

# ============================================
# PYTHONANYWHERE CONFIGURATION
# ============================================

# Add absolute path (CRITICAL)
# Replace 'yourusername' with your actual PythonAnywhere username
username = 'anamariascheau'  # <--- CHANGE THIS!
path = f'/home/{username}/mysite'
if path not in sys.path:
    sys.path.append(path)
    print(f"Added {path} to sys.path")

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Allow access from anywhere

# ============================================
# STEP 1: Load model and scaler with ABSOLUTE PATHS
# ============================================

# Use absolute paths for PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'enose_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'enose_scaler.pkl')
FEATURE_INFO_PATH = os.path.join(BASE_DIR, 'models', 'feature_info.pkl')

print("PythonAnywhere Deployment")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")

# Check if files exist
model = None
scaler = None
feature_info = None
feature_columns = []
classes = []

if os.path.exists(MODEL_PATH):
    try:
        print("Loading model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        
        feature_columns = feature_info['feature_columns']
        classes = feature_info['classes']
        
        print("Model loaded successfully!")
        print(f"   Features: {feature_columns}")
        print(f"   Classes: {classes}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model file not found at {MODEL_PATH}")
    print("   Will run in degraded mode (health check only)")

# ============================================
# STEP 2: Define API routes
# ============================================

@app.route('/', methods=['GET'])
def home():
    """Home page - checks if API is running"""
    return jsonify({
        'status': 'online',
        'message': 'Wine Detector API - PythonAnywhere',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Send data for prediction',
            '/health': 'GET - Check API status',
            '/info': 'GET - Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Check API status"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'pythonanywhere': True,
        'base_dir': BASE_DIR
    })

@app.route('/info', methods=['GET'])
def info():
    """Model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': str(type(model).__name__),
        'classes': list(classes),
        'feature_columns': feature_columns,
        'n_features': len(feature_columns),
        'n_classes': len(classes),
        'model_params': model.get_params() if hasattr(model, 'get_params') else {}
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives JSON with sensor data and returns prediction
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'help': 'Make sure model files are in /home/yourusername/mysite/models/'
        }), 503
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        print(f"Received: {data}")
        
        # Check if all required features are present
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Build DataFrame for prediction
        input_data = pd.DataFrame([{
            col: data[col] for col in feature_columns
        }])
        
        # Standardize
        input_scaled = scaler.transform(input_data[feature_columns])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Calculate confidence (highest probability)
        confidence = float(max(probabilities))
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'input_data': data,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                cls: float(prob) for cls, prob in zip(classes, probabilities)
            }
        }
        
        print(f"Response: {response['prediction']} (confidence: {confidence:.2%})")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for multiple predictions
    Receives a list of samples
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'Send JSON with "samples" key'}), 400
        
        samples = data['samples']
        results = []
        
        for sample in samples:
            # Check features
            missing = [f for f in feature_columns if f not in sample]
            if missing:
                results.append({
                    'error': f'Missing: {missing}',
                    'input': sample
                })
                continue
            
            # Build DataFrame
            input_df = pd.DataFrame([{col: sample[col] for col in feature_columns}])
            input_scaled = scaler.transform(input_df[feature_columns])
            
            # Prediction
            pred = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]
            
            results.append({
                'input': sample,
                'prediction': pred,
                'confidence': float(max(probs)),
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(classes, probs)
                }
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(samples),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# FOR PYTHONANYWHERE - DO NOT USE app.run() HERE!
# ============================================

# Leave this only for local testing - on PythonAnywhere this block is ignored
if __name__ == '__main__':
    print("\nStarting local server for testing...")
    print("   Access at: http://localhost:5000")
    print("   To stop: Ctrl+C\n")
    print("For PythonAnywhere, this block is ignored!")
    print("The WSGI server will use the 'app' object above.\n")
    
    # Run in debug mode for local testing
    app.run(host='0.0.0.0', port=5000, debug=True)