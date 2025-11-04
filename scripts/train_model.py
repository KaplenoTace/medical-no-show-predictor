#!/usr/bin/env python3
"""
Medical Appointment No-Show Predictor - Model Training Script

This script trains multiple ML models to predict whether a patient
will show up for their medical appointment based on historical data.

Author: Your Name
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='../data/appointments.csv'):
    """Load and prepare the dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    print("Preprocessing data...")
    # Add your preprocessing steps here
    # Example: handle missing values, encode categorical variables, etc.
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"F1 Score: {results[name]['f1']:.4f}")
        print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        # Save the best model
        if name == 'Random Forest':  # You can change this based on your results
            joblib.dump(model, '../models/best_model.pkl')
            print(f"\nSaved {name} as the best model.")
    
    return results

if __name__ == "__main__":
    print("Medical Appointment No-Show Prediction - Model Training")
    print("="*60)
    
    # Note: This is a starter template. You'll need to:
    # 1. Add actual data loading logic
    # 2. Implement proper preprocessing
    # 3. Feature engineering
    # 4. Hyperparameter tuning
    # 5. Cross-validation
    
    print("\nTo use this script:")
    print("1. Place your dataset in ../data/appointments.csv")
    print("2. Implement the preprocessing function")
    print("3. Run: python train_model.py")
    print("\nThis is a starter template for demonstration purposes.")
