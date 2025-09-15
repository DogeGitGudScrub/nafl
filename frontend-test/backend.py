#!/usr/bin/env python3
"""
Simple Backend Server for Liver Disease Prediction
Uses the trained XGBoost model directly from pickle files.
"""

import json
import joblib
import pandas as pd
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PredictionHandler(BaseHTTPRequestHandler):
    """HTTP request handler for predictions"""
    
    # Load model and bridge once when the server starts
    model = None
    bridge = None
    
    @classmethod
    def load_model(cls):
        """Load the trained model and data bridge"""
        try:
            print("Loading model and data bridge...")
            cls.model = joblib.load('../models/xgboost_liver_model.pkl')
            cls.bridge = joblib.load('../models/data_bridge.pkl')
            print("✅ Model and bridge loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            cls.model = None
            cls.bridge = None
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_status()
        elif self.path == '/health':
            self.serve_health()
        else:
            self.send_error(404, "Not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/predict':
            self.handle_prediction()
        else:
            self.send_error(404, "Not found")
    
    def serve_status(self):
        """Serve status page"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        status_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Liver Disease Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .status { background: #e8f5e8; padding: 15px; border-radius: 5px; }
                .error { background: #ffeaa7; padding: 15px; border-radius: 5px; }
                code { background: #f1f1f1; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🏥 Liver Disease Prediction API</h1>
        """
        
        if self.model is not None:
            status_html += """
            <div class="status">
                <h2>✅ Status: Ready</h2>
                <p>Model and data bridge loaded successfully!</p>
                <p><strong>Endpoints:</strong></p>
                <ul>
                    <li><code>POST /predict</code> - Make predictions</li>
                    <li><code>GET /health</code> - Health check</li>
                </ul>
            </div>
            """
        else:
            status_html += """
            <div class="error">
                <h2>❌ Status: Error</h2>
                <p>Failed to load model. Check that the model files exist in the models/ directory.</p>
            </div>
            """
        
        status_html += """
            <h3>Usage Example:</h3>
            <pre>
fetch('http://localhost:8080/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        age: 45,
        gender_male: 1,
        total_bilirubin: 2.5,
        // ... other features
    })
})
            </pre>
        </body>
        </html>
        """
        
        self.wfile.write(status_html.encode())
    
    def serve_health(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        health_status = {
            "status": "healthy" if self.model is not None else "error",
            "model_loaded": self.model is not None,
            "bridge_loaded": self.bridge is not None
        }
        
        self.wfile.write(json.dumps(health_status).encode())
    
    def handle_prediction(self):
        """Handle prediction requests"""
        try:
            # Check if model is loaded
            if self.model is None:
                self.send_error_response(500, "Model not loaded")
                return
            
            # Read request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Make prediction
            result = self.predict(data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
        except Exception as e:
            print(f"Prediction error: {e}")
            self.send_error_response(500, f"Prediction failed: {str(e)}")
    
    def predict(self, data):
        """Make prediction using the loaded model"""
        try:
            # Prepare the data in the format expected by the model
            # Create a DataFrame with all required features
            feature_data = self.prepare_features(data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(feature_data)[0, 1]
            prediction = self.model.predict(feature_data)[0]
            
            # Determine risk level
            if prediction_proba < 0.3:
                risk_level = 'low'
            elif prediction_proba < 0.7:
                risk_level = 'moderate'
            else:
                risk_level = 'high'
            
            # Get feature importance for this prediction
            feature_importance = self.get_feature_contributions(feature_data)
            
            return {
                'probability': float(prediction_proba),
                'prediction': int(prediction),
                'riskLevel': risk_level,
                'confidence': float(prediction_proba * 100),
                'keyFactors': self.get_key_factors(feature_data.iloc[0], prediction_proba),
                'featureImportance': feature_importance[:5]  # Top 5 features
            }
            
        except Exception as e:
            raise Exception(f"Prediction processing failed: {str(e)}")
    
    def prepare_features(self, data):
        """Prepare features in the format expected by the model"""
        features = {
            # Demographics
            'age': float(data.get('age', 40)),
            'gender_male': int(data.get('gender_male', 0)),

            # Lab values
            'total_bilirubin': float(data.get('total_bilirubin', 1.0)),
            'direct_bilirubin': float(data.get('direct_bilirubin', 0.3)),
            'alkaline_phosphatase': float(data.get('alkaline_phosphatase', 200)),
            'alt_sgpt': float(data.get('alt_sgpt', 30)),
            'ast_sgot': float(data.get('ast_sgot', 35)),
            'total_proteins': float(data.get('total_proteins', 7.0)),
            'albumin': float(data.get('albumin', 4.0)),

            # Derived feature
            'ag_ratio': float(data.get('albumin', 4.0)) / max(float(data.get('total_proteins', 7.0)) - float(data.get('albumin', 4.0)), 0.1),

            # Symptoms
            'has_fatigue': int(data.get('has_fatigue', 0)),
            'has_pain': int(data.get('has_pain', 0)),
            'has_jaundice': int(data.get('has_jaundice', 0)),
            'has_nausea': int(data.get('has_nausea', 0)),
            'has_itching': int(data.get('has_itching', 0)),
            'has_bleeding': int(data.get('has_bleeding', 0)),
            'has_edema': int(data.get('has_edema', 0)),

            # Risk factors
            'has_alcoholism': int(data.get('has_alcoholism', 0)),
            'has_diabetes': int(data.get('has_diabetes', 0)),
            'has_obesity': int(data.get('has_obesity', 0)),
            'has_hepatitis_history': int(data.get('has_hepatitis_history', 0)),
            'has_surgery_history': int(data.get('has_surgery_history', 0)),
        }

        # Derived clinical indicators
        features['bilirubin_elevated'] = 1 if features['total_bilirubin'] > 1.2 else 0
        features['enzymes_elevated'] = 1 if (features['alt_sgpt'] > 56 or features['ast_sgot'] > 40) else 0
        features['proteins_low'] = 1 if features['total_proteins'] < 6.0 else 0

        # Count symptoms and risk factors
        symptoms = [features['has_fatigue'], features['has_pain'], features['has_jaundice'],
                    features['has_nausea'], features['has_itching'], features['has_bleeding'], features['has_edema']]
        risk_factors = [features['has_alcoholism'], features['has_diabetes'], features['has_obesity'],
                        features['has_hepatitis_history'], features['has_surgery_history']]

        features['multiple_symptoms'] = 1 if sum(symptoms) >= 2 else 0
        features['high_risk_profile'] = 1 if sum(risk_factors) >= 2 else 0

        # Source indicators
        features['source_lpd'] = 1
        features['source_hepar'] = 0

        # Correct column order (from LiverDataBridge.unified_columns)
        correct_order = [
            'age', 'gender_male',
            'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphatase',
            'alt_sgpt', 'ast_sgot', 'total_proteins', 'albumin', 'ag_ratio',
            'has_fatigue', 'has_pain', 'has_jaundice', 'has_nausea',
            'has_itching', 'has_bleeding', 'has_edema',
            'has_alcoholism', 'has_diabetes', 'has_obesity',
            'has_hepatitis_history', 'has_surgery_history',
            'bilirubin_elevated', 'enzymes_elevated', 'proteins_low',
            'multiple_symptoms', 'high_risk_profile',
            'source_lpd', 'source_hepar'
        ]

        df = pd.DataFrame([features])
        df = df[correct_order]  # Reorder columns to match training
        return df
    
    def get_feature_contributions(self, feature_data):
        """Get feature importance for this specific prediction"""
        try:
            # Get feature importance from the model
            if hasattr(self.model, 'feature_importances_'):
                feature_names = feature_data.columns
                importances = self.model.feature_importances_
                
                feature_importance = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importances)
                ]
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance
            else:
                return []
        except:
            return []
    
    def get_key_factors(self, features, probability):
        """Identify key contributing factors for this prediction"""
        factors = []
        
        # Check lab values
        if features['total_bilirubin'] > 1.2:
            factors.append(f"Elevated Total Bilirubin ({features['total_bilirubin']:.1f} mg/dL)")
        if features['direct_bilirubin'] > 0.3:
            factors.append(f"Elevated Direct Bilirubin ({features['direct_bilirubin']:.1f} mg/dL)")
        if features['alt_sgpt'] > 56:
            factors.append(f"Elevated ALT ({features['alt_sgpt']:.0f} U/L)")
        if features['ast_sgot'] > 40:
            factors.append(f"Elevated AST ({features['ast_sgot']:.0f} U/L)")
        if features['albumin'] < 3.5:
            factors.append(f"Low Albumin ({features['albumin']:.1f} g/dL)")
        
        # Check symptoms
        if features['has_jaundice']:
            factors.append("Jaundice symptoms present")
        if features['has_fatigue']:
            factors.append("Fatigue reported")
        if features['has_pain']:
            factors.append("Abdominal pain reported")
        
        # Check risk factors
        if features['has_alcoholism']:
            factors.append("History of alcoholism")
        if features['has_hepatitis_history']:
            factors.append("Previous hepatitis infection")
        if features['has_diabetes']:
            factors.append("Diabetes mellitus")
        
        # Return top factors
        return factors[:5]
    
    def send_error_response(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            'error': message,
            'code': code
        }
        
        self.wfile.write(json.dumps(error_response).encode())


def run_server(port=8080):
    """Run the prediction server"""
    # Load the model when server starts
    PredictionHandler.load_model()
    
    # Create and start server
    server_address = ('', port)
    httpd = HTTPServer(server_address, PredictionHandler)
    
    print(f"🚀 Liver Disease Prediction Server starting...")
    print(f"📡 Server running on http://localhost:{port}")
    print(f"🏥 Model Status: {'Ready' if PredictionHandler.model else 'Error'}")
    print(f"💡 Open http://localhost:{port} in browser for status")
    print(f"🛑 Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped")
        httpd.shutdown()


if __name__ == '__main__':
    # Change to the frontend-test directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_server()
