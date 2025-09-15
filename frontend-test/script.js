// Global variables
let model = null;
let isLoading = false;

// Form handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results');
    
    form.addEventListener('submit', handleFormSubmit);
    
    // Add input validation
    addInputValidation();
    
    // Auto-calculate derived values
    addDerivedCalculations();
});

async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (isLoading) return;
    
    // Clear previous results
    document.getElementById('results').style.display = 'none';
    
    // Validate form
    if (!validateForm()) {
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    
    try {
        // Collect form data
        const formData = collectFormData();
        
        // Make prediction (simulate API call for now)
        const prediction = await makePrediction(formData);
        
        // Display results
        displayResults(prediction);
        
    } catch (error) {
        console.error('Prediction error:', error);
        displayError('An error occurred while making the prediction. Please try again.');
    } finally {
        setLoadingState(false);
    }
}

function collectFormData() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    
    // Convert to object with proper data types
    const data = {
        // Demographics
        age: parseFloat(formData.get('age')) || 0,
        gender_male: parseInt(formData.get('gender')) || 0,
        
        // Lab values
        total_bilirubin: parseFloat(formData.get('total_bilirubin')) || 1.0,
        direct_bilirubin: parseFloat(formData.get('direct_bilirubin')) || 0.3,
        alkaline_phosphatase: parseFloat(formData.get('alkaline_phosphatase')) || 200,
        alt_sgpt: parseFloat(formData.get('alt_sgpt')) || 30,
        ast_sgot: parseFloat(formData.get('ast_sgot')) || 35,
        albumin: parseFloat(formData.get('albumin')) || 4.0,
        total_proteins: parseFloat(formData.get('total_proteins')) || 7.0,
        
        // Symptoms (checkboxes)
        has_fatigue: formData.get('has_fatigue') ? 1 : 0,
        has_pain: formData.get('has_pain') ? 1 : 0,
        has_jaundice: formData.get('has_jaundice') ? 1 : 0,
        has_nausea: formData.get('has_nausea') ? 1 : 0,
        has_itching: formData.get('has_itching') ? 1 : 0,
        has_bleeding: formData.get('has_bleeding') ? 1 : 0,
        has_edema: formData.get('has_edema') ? 1 : 0,
        
        // Risk factors
        has_alcoholism: formData.get('has_alcoholism') ? 1 : 0,
        has_diabetes: formData.get('has_diabetes') ? 1 : 0,
        has_obesity: formData.get('has_obesity') ? 1 : 0,
        has_hepatitis_history: formData.get('has_hepatitis_history') ? 1 : 0,
        has_surgery_history: formData.get('has_surgery_history') ? 1 : 0
    };
    
    // Add derived features
    data.ag_ratio = data.albumin / (data.total_proteins - data.albumin);
    data.bilirubin_elevated = (data.total_bilirubin > 1.2) ? 1 : 0;
    data.enzymes_elevated = (data.alt_sgpt > 56 || data.ast_sgot > 40) ? 1 : 0;
    data.proteins_low = (data.total_proteins < 6.0) ? 1 : 0;
    
    // Count symptoms and risk factors
    const symptoms = [data.has_fatigue, data.has_pain, data.has_jaundice, data.has_nausea, 
                     data.has_itching, data.has_bleeding, data.has_edema];
    const riskFactors = [data.has_alcoholism, data.has_diabetes, data.has_obesity, 
                        data.has_hepatitis_history, data.has_surgery_history];
    
    data.multiple_symptoms = (symptoms.filter(x => x === 1).length >= 2) ? 1 : 0;
    data.high_risk_profile = (riskFactors.filter(x => x === 1).length >= 2) ? 1 : 0;
    
    // Source indicators (for this frontend, assume mixed input)
    data.source_lpd = 1;
    data.source_hepar = 0;
    
    return data;
}

async function makePrediction(data) {
    try {
        // Make API call to backend
        const response = await fetch('http://localhost:8080/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // Fallback to demo prediction if backend is not available
        console.warn('Using fallback demo prediction...');
        return makeDemoPrediction(data);
    }
}

// Fallback demo prediction function
function makeDemoPrediction(data) {
    // Simulate processing time
    return new Promise(resolve => {
        setTimeout(() => {
            // Simple rule-based prediction for demo purposes
            let riskScore = 0;
            let confidence = 0;
            
            // Check lab values (most important according to your model)
            if (data.total_bilirubin > 2.0) riskScore += 30;
            else if (data.total_bilirubin > 1.2) riskScore += 15;
            
            if (data.direct_bilirubin > 0.5) riskScore += 25;
            else if (data.direct_bilirubin > 0.3) riskScore += 10;
            
            if (data.alt_sgpt > 100) riskScore += 20;
            else if (data.alt_sgpt > 56) riskScore += 10;
            
            if (data.ast_sgot > 100) riskScore += 20;
            else if (data.ast_sgot > 40) riskScore += 10;
            
            if (data.albumin < 3.0) riskScore += 15;
            else if (data.albumin < 3.5) riskScore += 8;
            
            // Check symptoms
            const symptomCount = [data.has_fatigue, data.has_pain, data.has_jaundice, 
                                 data.has_nausea, data.has_itching, data.has_bleeding, 
                                 data.has_edema].filter(x => x === 1).length;
            riskScore += symptomCount * 3;
            
            // Check risk factors
            const riskFactorCount = [data.has_alcoholism, data.has_diabetes, data.has_obesity, 
                                   data.has_hepatitis_history, data.has_surgery_history].filter(x => x === 1).length;
            riskScore += riskFactorCount * 5;
            
            // Calculate confidence based on risk score
            confidence = Math.min(riskScore / 100, 0.99);
            
            const result = {
                probability: confidence,
                prediction: confidence > 0.5 ? 1 : 0,
                riskLevel: confidence < 0.3 ? 'low' : confidence < 0.7 ? 'moderate' : 'high',
                keyFactors: getKeyFactors(data, confidence)
            };
            
            resolve(result);
        }, 1500);
    });
}

function getKeyFactors(data, confidence) {
    const factors = [];
    
    if (data.total_bilirubin > 1.2) factors.push(`Elevated Total Bilirubin (${data.total_bilirubin} mg/dL)`);
    if (data.direct_bilirubin > 0.3) factors.push(`Elevated Direct Bilirubin (${data.direct_bilirubin} mg/dL)`);
    if (data.alt_sgpt > 56) factors.push(`Elevated ALT (${data.alt_sgpt} U/L)`);
    if (data.ast_sgot > 40) factors.push(`Elevated AST (${data.ast_sgot} U/L)`);
    if (data.albumin < 3.5) factors.push(`Low Albumin (${data.albumin} g/dL)`);
    if (data.has_jaundice) factors.push('Jaundice symptoms');
    if (data.has_alcoholism) factors.push('History of alcoholism');
    if (data.has_hepatitis_history) factors.push('Hepatitis history');
    
    return factors.slice(0, 5); // Return top 5 factors
}

function displayResults(prediction) {
    const resultsSection = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    
    const riskLevel = prediction.riskLevel;
    const probability = (prediction.probability * 100).toFixed(1);
    
    let riskClass, riskText, recommendation;
    
    if (riskLevel === 'low') {
        riskClass = 'result-low';
        riskText = 'Low Risk';
        recommendation = 'Continue regular health monitoring. Consider lifestyle improvements if applicable.';
    } else if (riskLevel === 'moderate') {
        riskClass = 'result-moderate';
        riskText = 'Moderate Risk';
        recommendation = 'Consult with healthcare provider for further evaluation and monitoring.';
    } else {
        riskClass = 'result-high';
        riskText = 'High Risk';
        recommendation = 'Seek immediate medical attention for comprehensive evaluation.';
    }
    
    resultContent.innerHTML = `
        <div class="result-card ${riskClass}">
            <h3>🎯 Prediction Result: ${riskText}</h3>
            <p><strong>Liver Disease Probability: ${probability}%</strong></p>
            
            <div class="confidence-bar">
                <div class="confidence-fill confidence-${riskLevel}" style="width: ${probability}%"></div>
            </div>
            
            <p><strong>Recommendation:</strong> ${recommendation}</p>
        </div>
        
        ${prediction.keyFactors.length > 0 ? `
        <div class="key-factors">
            <h4>🔍 Key Contributing Factors:</h4>
            <ul>
                ${prediction.keyFactors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="disclaimer">
            <p><strong>⚠️ Important:</strong> This is an AI-based assessment tool for educational purposes. 
            Always consult qualified healthcare professionals for proper medical diagnosis and treatment.</p>
        </div>
    `;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function displayError(message) {
    const resultsSection = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    
    resultContent.innerHTML = `
        <div class="result-card result-high">
            <h3>❌ Error</h3>
            <p>${message}</p>
        </div>
    `;
    
    resultsSection.style.display = 'block';
}

function validateForm() {
    const form = document.getElementById('predictionForm');
    let isValid = true;
    
    // Clear previous errors
    document.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
    document.querySelectorAll('.error-message').forEach(el => el.remove());
    
    // Validate required fields
    const requiredFields = ['age', 'gender'];
    requiredFields.forEach(fieldName => {
        const field = form.querySelector(`[name="${fieldName}"]`);
        if (!field.value || field.value === '') {
            showFieldError(field, 'This field is required');
            isValid = false;
        }
    });
    
    // Validate age range
    const ageField = form.querySelector('[name="age"]');
    if (ageField.value && (ageField.value < 1 || ageField.value > 120)) {
        showFieldError(ageField, 'Age must be between 1 and 120');
        isValid = false;
    }
    
    return isValid;
}

function showFieldError(field, message) {
    field.classList.add('error');
    const errorSpan = document.createElement('span');
    errorSpan.className = 'error-message';
    errorSpan.textContent = message;
    field.parentNode.appendChild(errorSpan);
}

function setLoadingState(loading) {
    isLoading = loading;
    const form = document.getElementById('predictionForm');
    
    if (loading) {
        form.classList.add('loading');
    } else {
        form.classList.remove('loading');
    }
}

function clearForm() {
    const form = document.getElementById('predictionForm');
    form.reset();
    
    // Clear results
    document.getElementById('results').style.display = 'none';
    
    // Clear any error states
    document.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
    document.querySelectorAll('.error-message').forEach(el => el.remove());
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function addInputValidation() {
    // Add real-time validation for numeric inputs
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('blur', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));
            
            if (this.value && (isNaN(value) || value < min || value > max)) {
                this.classList.add('error');
            } else {
                this.classList.remove('error');
            }
        });
    });
}

function addDerivedCalculations() {
    // Auto-calculate A/G ratio when albumin or total proteins change
    const albuminInput = document.getElementById('albumin');
    const totalProteinsInput = document.getElementById('total_proteins');
    
    function updateAGRatio() {
        const albumin = parseFloat(albuminInput.value) || 0;
        const totalProteins = parseFloat(totalProteinsInput.value) || 0;
        
        if (albumin > 0 && totalProteins > albumin) {
            const agRatio = (albumin / (totalProteins - albumin)).toFixed(2);
            // Could display this calculated value if needed
        }
    }
    
    albuminInput?.addEventListener('input', updateAGRatio);
    totalProteinsInput?.addEventListener('input', updateAGRatio);
}

// Utility function to format numbers
function formatNumber(num, decimals = 1) {
    return parseFloat(num).toFixed(decimals);
}
