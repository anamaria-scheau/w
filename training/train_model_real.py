"""
Training Script for Wine Detector
This script trains a KNN model using real sensor data from a CSV file.
Place your data in 'data/bme688_readings.csv' and run this script.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

def main():
    print("=" * 60)
    print("WINE DETECTOR - MODEL TRAINING")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load data
    data_path = 'data/bme688_readings.csv'
    
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("\nPlease create a CSV file with the following structure:")
        print("temperature,humidity,gas_resistance,iaq,label")
        print("23.4,66.2,82345,118,red_wine")
        print("22.1,44.8,205678,24,air")
        print("23.2,71.5,118234,93,white_wine")
        return
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total samples: {len(df)}")
    print(f"Classes found: {sorted(df['label'].unique())}")
    
    # Prepare features and target
    feature_columns = ['temperature', 'humidity', 'gas_resistance', 'iaq']
    X = df[feature_columns]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find best K value
    print("\nTesting different K values...")
    k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    best_score = 0
    best_k = 5
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        print(f"  K={k}: accuracy = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    
    print(f"\nBest K value: {best_k} (accuracy: {best_score:.3f})")
    
    # Detailed evaluation
    print("\nClassification Report:")
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    # Save model and artifacts
    print("\nSaving model and associated files...")
    
    model_path = 'models/enose_model.pkl'
    scaler_path = 'models/enose_scaler.pkl'
    info_path = 'models/feature_info.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    feature_info = {
        'feature_columns': feature_columns,
        'classes': list(best_model.classes_)
    }
    joblib.dump(feature_info, info_path)
    
    print(f"  Model saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")
    print(f"  Feature info saved: {info_path}")
    
    # Quick validation
    print("\nValidating with average sample...")
    sample = pd.DataFrame([{
        'temperature': df['temperature'].mean(),
        'humidity': df['humidity'].mean(),
        'gas_resistance': df['gas_resistance'].mean(),
        'iaq': df['iaq'].mean()
    }])
    
    sample_scaled = scaler.transform(sample[feature_columns])
    prediction = best_model.predict(sample_scaled)[0]
    probabilities = best_model.predict_proba(sample_scaled)[0]
    
    print(f"  Average sample prediction: {prediction}")
    print("  Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"    {best_model.classes_[i]}: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API: python app.py")
    print("2. Test with ESP32 or dashboard")

if __name__ == "__main__":
    main()