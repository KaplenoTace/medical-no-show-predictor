#!/usr/bin/env python3
"""
Medical Appointment No-Show Predictor - Frontend Application

This Flask application provides a web interface for predicting
whether a patient will show up for their medical appointment.

Author: Your Name
Date: November 2025
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model (update path as needed)
# model = joblib.load('../models/best_model.pkl')

@app.route('/')
def home():
    """Render the home page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical No-Show Predictor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .form-group {
                margin: 15px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 4px;
                display: none;
            }
            .show {
                background-color: #d4edda;
                color: #155724;
            }
            .no-show {
                background-color: #f8d7da;
                color: #721c24;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical Appointment No-Show Predictor</h1>
            <p>Enter patient information to predict appointment attendance.</p>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Age:</label>
                    <input type="number" id="age" name="age" required>
                </div>
                
                <div class="form-group">
                    <label>Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select...</option>
                        <option value="M">Male</option>
                        <option value="F">Female</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Days Until Appointment:</label>
                    <input type="number" id="days" name="days" required>
                </div>
                
                <button type="submit">Predict</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            // This is a placeholder for demonstration
            // In production, this would call the backend API
            document.getElementById('predictionForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Mock prediction (replace with actual API call)
                const resultDiv = document.getElementById('result');
                const mockPrediction = Math.random() > 0.5;
                
                if (mockPrediction) {
                    resultDiv.className = 'show';
                    resultDiv.innerHTML = '<h3>✅ Prediction: Patient will SHOW UP</h3><p>Confidence: 85%</p>';
                } else {
                    resultDiv.className = 'no-show';
                    resultDiv.innerHTML = '<h3>⚠️ Prediction: Patient will NOT SHOW UP</h3><p>Confidence: 78%</p>';
                }
                
                resultDiv.style.display = 'block';
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        data = request.json
        
        # Placeholder for actual prediction logic
        # In production, you would:
        # 1. Process the input data
        # 2. Load the model
        # 3. Make prediction
        # 4. Return results
        
        prediction = {
            'prediction': 'show',  # or 'no-show'
            'confidence': 0.85,
            'message': 'This is a placeholder. Implement model prediction logic here.'
        }
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Medical No-Show Predictor - Frontend Application")
    print("="*60)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nNote: This is a starter template for demonstration.")
    print("To use in production:")
    print("1. Implement actual model loading")
    print("2. Add proper input validation")
    print("3. Configure for production deployment")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
