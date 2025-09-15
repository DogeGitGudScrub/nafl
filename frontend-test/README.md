# Liver Disease Prediction System

A complete web-based liver disease prediction system using XGBoost machine learning model with 99.9% AUC performance.

## System Overview

This system combines two datasets (LPD and HEPAR) to create a unified liver disease prediction model. It features:

- **Machine Learning Backend**: XGBoost classifier trained on combined datasets
- **Web Frontend**: Professional medical-grade interface
- **Real-time Predictions**: Direct model integration without Flask overhead

## File Structure

```
frontend-test/
├── index.html          # Main web interface
├── styles.css          # Professional medical styling
├── script.js           # Frontend logic with API integration
├── backend.py          # Python prediction server
├── start_server.bat    # Windows batch script to start server
└── README.md          # This file

models/
├── xgboost_liver_model.pkl    # Trained XGBoost model
├── data_bridge.pkl            # Data preprocessing bridge
└── model_metadata.json       # Model performance metrics
```

## Quick Start

### 1. Start the Backend Server

**Option A: Using the batch script (Windows)**
```cmd
cd frontend-test
start_server.bat
```

**Option B: Manual start**
```cmd
cd frontend-test
python backend.py
```

### 2. Open the Web Interface

- **Backend Status**: Open http://localhost:8080 in your browser
- **Web Interface**: Open `index.html` in your browser

### 3. Make Predictions

1. Fill out the patient information form
2. Enter lab values, symptoms, and risk factors
3. Click "Analyze Patient Data"
4. View the prediction results and risk assessment

## Backend API

### Health Check
```
GET http://localhost:8080/health
```

### Make Prediction
```
POST http://localhost:8080/predict
Content-Type: application/json

{
    "age": 45,
    "gender_male": 1,
    "total_bilirubin": 2.5,
    "direct_bilirubin": 0.8,
    "alkaline_phosphatase": 250,
    "alt_sgpt": 85,
    "ast_sgot": 78,
    "albumin": 3.2,
    "total_proteins": 6.8,
    "has_fatigue": 1,
    "has_jaundice": 1,
    "has_alcoholism": 0,
    ...
}
```

### Response Format
```json
{
    "probability": 0.87,
    "prediction": 1,
    "riskLevel": "high",
    "confidence": 87.0,
    "keyFactors": [
        "Elevated Total Bilirubin (2.5 mg/dL)",
        "Elevated Direct Bilirubin (0.8 mg/dL)",
        "Low Albumin (3.2 g/dL)",
        "Jaundice symptoms present",
        "Fatigue reported"
    ],
    "featureImportance": [...]
}
```

## Model Information

- **Algorithm**: XGBoost Classifier
- **Performance**: 99.9% AUC on test data
- **Features**: 29 clinical features including demographics, lab values, symptoms, and risk factors
- **Datasets**: Combined LPD (30,000 samples) and HEPAR (500 samples)

### Key Features Used
- **Demographics**: Age, Gender
- **Lab Values**: Bilirubin levels, liver enzymes, proteins
- **Symptoms**: Fatigue, jaundice, pain, nausea, etc.
- **Risk Factors**: Alcoholism, diabetes, hepatitis history, etc.

## Troubleshooting

### Backend Won't Start
- Ensure Python is installed and in PATH
- Check that all required packages are installed: `pip install scikit-learn xgboost pandas joblib`
- Verify model files exist in `../models/` directory

### Predictions Fail
- Check that backend server is running at http://localhost:8080
- Verify the `/health` endpoint returns "healthy" status
- Look at browser console for error messages

### CORS Issues
- The backend includes CORS headers for local development
- For production, configure proper CORS policies

## Development Notes

### Fallback Mode
If the backend is unavailable, the frontend automatically falls back to a demo prediction mode using rule-based logic.

### Model Retraining
To retrain the model:
1. Run the `training.ipynb` notebook
2. Ensure new model files are saved to `models/` directory
3. Restart the backend server

### Customization
- Modify `backend.py` to add new features or change prediction logic
- Update `script.js` to change frontend behavior
- Customize `styles.css` for different styling

## Production Deployment

For production use:
1. Use a proper WSGI server (gunicorn, uwsgi)
2. Implement proper authentication and authorization
3. Add input validation and sanitization
4. Set up proper logging and monitoring
5. Use HTTPS for secure communication

## License

This system is for educational and research purposes. Ensure compliance with medical data regulations in your jurisdiction.
├── script.js           # JavaScript functionality and form handling
└── README.md           # This file
```

## Features

### 🎨 **Modern UI Design**
- Responsive design that works on desktop and mobile
- Gradient backgrounds and glass-morphism effects
- Clean, medical-grade interface
- Smooth animations and transitions

### 📝 **Comprehensive Form**
- **Demographics**: Age, Gender
- **Lab Values**: Bilirubin, ALT, AST, Albumin, etc. with normal ranges
- **Symptoms**: Checkboxes for fatigue, pain, jaundice, etc.
- **Risk Factors**: Alcoholism, diabetes, obesity, etc.

### 🔬 **Smart Features**
- Real-time form validation
- Auto-calculation of derived values (A/G ratio)
- Visual feedback for elevated values
- Loading states and error handling

### 📊 **Results Display**
- Color-coded risk levels (Low/Moderate/High)
- Confidence percentage with visual bar
- Key contributing factors
- Medical recommendations
- Clear disclaimers

## How to Use

1. **Open the Interface**
   ```bash
   # Simply open index.html in a web browser
   # Or serve it with a local server:
   python -m http.server 8000
   # Then visit: http://localhost:8000
   ```

2. **Fill the Form**
   - Enter patient demographics
   - Input available lab values (normal ranges shown)
   - Check relevant symptoms
   - Select applicable risk factors

3. **Get Prediction**
   - Click "🔬 Analyze Risk"
   - View color-coded results
   - Review contributing factors
   - Follow medical recommendations

## Technical Details

### Form Fields Match Model Features
The form includes all key features from your XGBoost model:
- Demographics: `age`, `gender_male`
- Lab Values: `total_bilirubin`, `direct_bilirubin`, `alt_sgpt`, `ast_sgot`, `albumin`, etc.
- Symptoms: `has_fatigue`, `has_pain`, `has_jaundice`, etc.
- Risk Factors: `has_alcoholism`, `has_diabetes`, etc.

### Current Implementation
- **Frontend Only**: Currently uses JavaScript-based prediction simulation
- **Demo Logic**: Simple rule-based scoring for demonstration
- **Real Implementation**: Ready to connect to Python backend API

## Integration with Your Model

To connect this frontend to your trained XGBoost model:

### Option 1: Flask/FastAPI Backend
```python
# Create a simple API endpoint
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('models/xgboost_liver_model.pkl')
bridge = joblib.load('models/data_bridge.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert to DataFrame
    df = pd.DataFrame([data])
    # Make prediction
    probability = model.predict_proba(df)[0, 1]
    prediction = model.predict(df)[0]
    
    return jsonify({
        'probability': float(probability),
        'prediction': int(prediction),
        'riskLevel': 'high' if probability > 0.7 else 'moderate' if probability > 0.3 else 'low'
    })
```

### Option 2: Modify JavaScript
Update the `makePrediction()` function in `script.js`:
```javascript
async function makePrediction(data) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return await response.json();
}
```

## Responsive Design

The interface adapts to different screen sizes:
- **Desktop**: Multi-column layout with full features
- **Tablet**: Optimized grid layout
- **Mobile**: Single-column, touch-friendly interface

## Medical Compliance

- ⚠️ Clear disclaimers about educational use only
- 🏥 Recommendations to consult healthcare professionals
- 📋 Professional medical terminology and units
- 🔒 No data storage or transmission (privacy-focused)

## Browser Support

Compatible with all modern browsers:
- Chrome, Firefox, Safari, Edge
- Mobile browsers (iOS Safari, Chrome Mobile)
- Requires JavaScript enabled

---

**Ready to Use**: Simply open `index.html` in a browser to start testing the interface!
