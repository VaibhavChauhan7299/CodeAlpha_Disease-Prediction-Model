import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "best_model.pkl"
preprocessor_path = BASE_DIR / "models" / "preprocessor.pkl"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


# ==========================================
# HEADER
# ==========================================

st.title("❤️ Heart Disease Risk Prediction")

st.markdown(
    """
    Use this machine learning application to estimate the probability
    of heart disease based on selected patient health information.
    """
)

st.warning(
    "⚠️ This application is for educational and research purposes only. "
    "It is not a medical diagnosis and should not replace professional "
    "medical advice."
)


# ==========================================
# PATIENT INFORMATION
# ==========================================

st.header("👤 Patient Information")

st.markdown(
    "Enter the patient's health information below."
)

col1, col2, col3 = st.columns(3)


# ==========================================
# COLUMN 1
# ==========================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55,
        step=1,
        help="Patient's age in years."
    )

    sex_label = st.selectbox(
        "Sex",
        ["Female", "Male"],
        help="Biological sex recorded in the dataset."
    )

    sex = 0 if sex_label == "Female" else 1

    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ],
        help="Type of chest pain experienced by the patient."
    )

    cp_mapping = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4
    }

    cp = cp_mapping[cp_label]

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80.0,
        max_value=250.0,
        value=140.0,
        step=1.0,
        help="Resting blood pressure in mm Hg."
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100.0,
        max_value=600.0,
        value=250.0,
        step=1.0,
        help="Serum cholesterol level in mg/dl."
    )


# ==========================================
# COLUMN 2
# ==========================================

with col2:

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"],
        help="Whether fasting blood sugar is greater than 120 mg/dl."
    )

    fbs = 0 if fbs_label == "No" else 1

    restecg_label = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ],
        help="Resting electrocardiographic result."
    )

    restecg_mapping = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    restecg = restecg_mapping[restecg_label]

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50.0,
        max_value=250.0,
        value=150.0,
        step=1.0,
        help="Maximum heart rate achieved during the test."
    )

    exang_label = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="Whether exercise caused angina."
    )

    exang = 0 if exang_label == "No" else 1

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.2,
        step=0.1,
        help="ST depression induced by exercise relative to rest."
    )


# ==========================================
# COLUMN 3
# ==========================================

with col3:

    slope_label = st.selectbox(
        "ST Segment Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ],
        help="Slope of the peak exercise ST segment."
    )

    slope_mapping = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }

    slope = slope_mapping[slope_label]

    ca = st.selectbox(
        "Number of Major Vessels (CA)",
        [0, 1, 2, 3],
        help="Number of major vessels colored by fluoroscopy."
    )

    thal_label = st.selectbox(
        "Thalassemia (Thal)",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ],
        help="Thalassemia-related categorical measurement."
    )

    thal_mapping = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    }

    thal = thal_mapping[thal_label]


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    type="primary",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    patient_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    patient_processed = preprocessor.transform(patient_data)

    prediction = model.predict(patient_processed)[0]

    probability = model.predict_proba(
        patient_processed
    )[0][1]

    probability_percent = probability * 100


    # ==========================================
    # RESULT
    # ==========================================

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Potential Heart Disease Risk Detected"
        )

    else:

        st.success(
            "✅ No Heart Disease Risk Detected"
        )


    # ==========================================
    # PROBABILITY
    # ==========================================

    st.subheader("Estimated Heart Disease Probability")

    st.metric(
        label="Model Estimated Probability",
        value=f"{probability_percent:.2f}%"
    )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    if probability_percent < 30:

        st.info(
            "The model estimates a relatively low probability "
            "for the positive class."
        )

    elif probability_percent < 70:

        st.warning(
            "The model estimates an intermediate probability "
            "for the positive class."
        )

    else:

        st.error(
            "The model estimates a relatively high probability "
            "for the positive class."
        )


    # ==========================================
    # INPUT SUMMARY
    # ==========================================

    with st.expander("📋 View Entered Patient Information"):

        display_data = {
            "Age": age,
            "Sex": sex_label,
            "Chest Pain Type": cp_label,
            "Resting Blood Pressure": trestbps,
            "Cholesterol": chol,
            "Fasting Blood Sugar > 120": fbs_label,
            "Resting ECG": restecg_label,
            "Maximum Heart Rate": thalach,
            "Exercise Induced Angina": exang_label,
            "ST Depression": oldpeak,
            "ST Segment Slope": slope_label,
            "Major Vessels (CA)": ca,
            "Thalassemia": thal_label
        }

        st.table(
            pd.DataFrame(
                display_data.items(),
                columns=["Feature", "Value"]
            )
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Heart Disease Risk Prediction • Machine Learning Project"
)

st.caption(
    "Educational / research demonstration only. "
    "This model is not clinically validated."
)