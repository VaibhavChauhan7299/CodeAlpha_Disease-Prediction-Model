import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL AND PREPROCESSOR
# ==========================================

model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    "Enter the patient's information below to estimate "
    "heart disease risk using a machine learning model."
)

st.warning(
    "⚠️ This application is for educational and research purposes only. "
    "It is not a medical diagnosis or a substitute for professional medical advice."
)


# ==========================================
# PATIENT INFORMATION
# ==========================================

st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80.0,
        max_value=250.0,
        value=140.0
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100.0,
        max_value=600.0,
        value=250.0
    )

with col2:
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50.0,
        max_value=250.0,
        value=150.0
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.2
    )

with col3:
    slope = st.selectbox(
        "Slope",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels (CA)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        options=[3, 6, 7]
    )


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Predict Heart Disease Risk", type="primary"):

    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    patient_df = pd.DataFrame([patient_data])

    # Preprocess patient data
    patient_processed = preprocessor.transform(patient_df)

    # Prediction
    prediction = model.predict(patient_processed)[0]

    # Probability
    probability = model.predict_proba(
        patient_processed
    )[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Heart Disease Risk Detected")
    else:
        st.success("✅ No Heart Disease Risk Detected")

    st.metric(
        "Estimated Heart Disease Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.info(
        "The probability shown is the model's estimated probability "
        "for the positive class based on the training dataset."
    )