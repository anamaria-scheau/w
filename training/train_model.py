"""
Training Script for Wine Detector
This script generates simulated data to test the entire pipeline.
When real data becomes available, only the 'generate_simulated_data()' function needs to be replaced.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# STEP 1: Generate simulated data (replace with real data later)
# ============================================

def generate_simulated_data(n_samples_per_class=200):
    """
    Generate simulated data for 3 classes:
    - class 0: Air
    - class 1: Red wine
    - class 2: White wine
    
    Values are realistic for a BME688 sensor
    """
    np.random.seed(42)  # For reproducibility
    
    # Parameters for each class
    classes = {
        'air': {
            'temp': (22, 2),        # (mean, standard deviation)
            'hum': (45, 5),
            'gas': (200000, 30000),
            'iaq': (25, 10)
        },
        'red_wine': {
            'temp': (23, 1.5),
            'hum': (65, 8),
            'gas': (80000, 20000),
            'iaq': (120, 20)
        },
        'white_wine': {
            'temp': (23, 1.5),
            'hum': (70, 7),
            'gas': (120000, 25000),
            'iaq': (90, 15)
        }
    }
    
    data = []
    labels = []
    
    for class_name, params in classes.items():
        for _ in range(n_samples_per_class):
            sample = {
                'temperature': np.random.normal(params['temp'][0], params['temp'][1]),
                'humidity': np.random.normal(params['hum'][0], params['hum'][1]),
                'gas_resistance': np.random.normal(params['gas'][0], params['gas'][1]),
                'iaq': np.random.normal(params['iaq'][0], params['iaq'][1])
            }
            data.append(sample)
            labels.append(class_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    return df

print("Generating simulated data for testing...")
df = generate_simulated_data(n_samples_per_class=200)
print(f"Data generated: {len(df)} samples")
print(f"Classes: {df['label'].unique()}")

# ============================================
# STEP 2: Data exploration (optional)
# ============================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDescriptive statistics:")
print(df.groupby('label').describe())

# ============================================
# STEP 3: Prepare data for training
# ============================================

# Separate features (X) from target (y)
feature_columns = ['temperature', 'humidity', 'gas_resistance', 'iaq']
X = df[feature_columns]
y = df['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 4: Train KNN model
# ============================================

print("\nTraining KNN model (K=17)...")
knn_model = KNeighborsClassifier(
    n_neighbors=17,
    weights='uniform',
    metric='euclidean',
    p=2  # Euclidean distance
)

knn_model.fit(X_train_scaled, y_train)

# ============================================
# STEP 5: Evaluate the model
# ============================================

# Predictions on test set
y_pred = knn_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = knn_model.score(X_test_scaled, y_test)
print(f"\nTest set accuracy: {accuracy:.2%}")

# Detailed classification report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=knn_model.classes_,
            yticklabels=knn_model.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.show()

# ============================================
# STEP 6: Save model and scaler
# ============================================

print("\nSaving model and scaler...")

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save KNN model
joblib.dump(knn_model, 'models/enose_model.pkl')
print("Model saved: models/enose_model.pkl")

# Save scaler
joblib.dump(scaler, 'models/enose_scaler.pkl')
print("Scaler saved: models/enose_scaler.pkl")

# Save feature list (important for API)
feature_info = {
    'feature_columns': feature_columns,
    'classes': list(knn_model.classes_)
}
joblib.dump(feature_info, 'models/feature_info.pkl')
print("Feature info saved: models/feature_info.pkl")

# ============================================
# STEP 7: Test the saved model
# ============================================

print("\nTesting the saved model...")

# Load the model
loaded_model = joblib.load('models/enose_model.pkl')
loaded_scaler = joblib.load('models/enose_scaler.pkl')

# Test with a new sample
test_sample = pd.DataFrame([{
    'temperature': 23.2,
    'humidity': 68.5,
    'gas_resistance': 95000,
    'iaq': 110
}])

test_scaled = loaded_scaler.transform(test_sample[feature_columns])
prediction = loaded_model.predict(test_scaled)[0]
probabilities = loaded_model.predict_proba(test_scaled)[0]

print(f"Test sample: {test_sample.to_dict('records')[0]}")
print(f"Prediction: {prediction}")
print("Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"   {loaded_model.classes_[i]}: {prob:.2%}")

print("\nModel is functioning correctly!")
print("When real data becomes available, replace the generate_simulated_data() function")