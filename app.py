import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction AI",
    page_icon="https://img.icons8.com/?size=100&id=80705&format=png&color=000000",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting custom CSS for an elegant, modern, and dark-themed healthcare UI
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 15px 0px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        transform: scale(1.02);
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    h3 {
        color: #ecf0f1;
    }
    .metric-card {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d313f;
        margin-bottom: 15px;
    }
    .header-text {
        font-size: 18px;
        color: #a0a5c0;
    }
    </style>
""", unsafe_allow_html=True)

# Safely load the saved models and scalers
try:
    model = joblib.load('knn_heart.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('cols.pkl')
except FileNotFoundError:
    # Fallback in case of lowercase/uppercase mismatch in file names
    try:
        model = joblib.load('KNN_HEART.pkl')
        scaler = joblib.load('Scaler.pkl')
        model_columns = joblib.load('cols.pkl')
    except Exception as e:
        st.error(f"Error loading models or setup files: {e}")

# Layout definition
st.title("Heart Disease Prediction AI")
st.markdown("<p class='header-text'>Clinical Assistant & Heart Disease Prognosis Tool powered by Machine Learning</p>", unsafe_allow_html=True)
st.markdown("--- ")

# Split main view into columns
col_info, col_form = st.columns([1, 2])

with col_info:
    st.markdown("### About the Platform")
    st.write("CardioGuard AI implements a custom-tuned K-Nearest Neighbors classification algorithm to analyze patient clinical measurements and determine statistical risk levels for potential cardiovascular conditions.")
    
    st.markdown("<div class='metric-card'><h4>Clinical Guidelines</h4><ul><li>Ensure Resting Blood Pressure and Cholesterol values are verified.</li><li>Check the ST Slope orientation on the ECG report.</li></ul></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-card'><h4>Disclaimer</h4>This platform serves strictly as an educational clinical assistant and does not substitute professional medical diagnosis.</div>", unsafe_allow_html=True)

with col_form:
    st.markdown("### Patient Demographics and Measurements")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Patient Age", 1, 120, 45)
        sex = st.radio("Sex at Birth", ["Male", "Female"], horizontal=True)
        chest_pain = st.selectbox("Chest Pain Type", ["ASY: Asymptomatic", "NAP: Non-Anginal", "ATA: Atypical Angina", "TA: Typical Angina"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)

    with c2:
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], horizontal=True)
        resting_ecg = st.selectbox("Resting Electrocardiogram", ["Normal", "LVH: Left Ventricular Hypertrophy", "ST: ST-T wave abnormality"])
        max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exercise_angina = st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True)
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        st_slope = st.selectbox("Peak Exercise ST Segment Slope", ["Up", "Flat", "Down"])

    # Preprocess inputs when submit button is pressed
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("ANALYZE CARDIAC METRICS"):
        
        # Prepare categorical input matching training format
        sex_val = 'M' if sex == "Male" else 'F'
        cp_val = chest_pain.split(':')[0]
        fbs_val = 1 if fasting_bs == "Yes" else 0
        ecg_val = resting_ecg.split(':')[0]
        angina_val = 'Y' if exercise_angina == "Yes" else 'N'
        st_val = st_slope
        
        raw_data = {
            'Age': age,
            'Sex': sex_val,
            'ChestPainType': cp_val,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fbs_val,
            'RestingECG': ecg_val,
            'MaxHR': max_hr,
            'ExerciseAngina': angina_val,
            'Oldpeak': oldpeak,
            'ST_Slope': st_val
        }
        
        input_df = pd.DataFrame([raw_data])
        
        # Build encoded columns correctly matching training schema
        # Added dtype=int to prevent boolean/numeric mismatches in newer pandas versions
        input_encoded = pd.get_dummies(input_df, dtype=int)
        
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
                
        # Ensure column order strictly matches training before scaling
        input_encoded = input_encoded[model_columns]
        
        # Apply scaler to the ENTIRE dataframe (fixing the ValueError)
        scaled_array = scaler.transform(input_encoded)
        input_scaled = pd.DataFrame(scaled_array, columns=model_columns)
        
        # Process Prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        risk_prob = probabilities[1]
        
        st.markdown("--- ")
        st.markdown("### Diagnostic Assessment Output")
        
        if prediction == 1:
            st.error(f"ALERT: High risk indicators detected. Risk Score: {risk_prob:.1%}. Medical evaluation recommended.")
        else:
            st.success(f"NORMAL: Measurements appear stable. Risk Score: {risk_prob:.1%}. Keep maintaining a healthy routine.")