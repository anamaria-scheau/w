# wine_detector_wsgi.py
# This file should be located in /home/yourusername/mysite/

import sys
import os

# ============================================
# PYTHONANYWHERE WSGI CONFIGURATION
# ============================================

# Add project path to Python path - REPLACE 'yourusername' with your actual username
project_path = '/home/anamariascheau/mysite'
if project_path not in sys.path:
    sys.path.append(project_path)
    print(f"Added {project_path} to system path")

# Set environment variable to indicate we are on PythonAnywhere
os.environ['PYTHONANYWHERE'] = 'True'

# ============================================
# IMPORT FLASK APPLICATION
# ============================================
# Import the Flask app from api.py
# The modified API exports 'app' - exactly what PythonAnywhere needs
from api import app as app

# ============================================
# LOGGING CONFIGURATION (Optional)
# ============================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================
# STARTUP VERIFICATION
# ============================================
print("=" * 50)
print("Wine Detector API starting on PythonAnywhere")
print(f"Project path: {project_path}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print("=" * 50)

# Optional: Verify that model files exist
models_dir = os.path.join(project_path, 'models')
if os.path.exists(models_dir):
    print(f"Models directory found: {models_dir}")
    model_files = os.listdir(models_dir)
    print(f"Model files: {model_files}")
else:
    print(f"WARNING: Models directory not found at {models_dir}")

print("WSGI initialization complete")