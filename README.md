# ❤️ Disease Prediction Model

A machine learning-based web application that predicts the risk of heart disease using patient medical information. The project compares multiple machine learning algorithms, performs hyperparameter tuning and cross-validation, and provides predictions through an interactive Streamlit application.

> **⚠️ Disclaimer:** This project is developed for educational and research purposes only. It is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://codealphadisease-prediction-model-zrqups5zo7qh3f8ctyofp9.streamlit.app/

The application allows users to enter patient information and receive a predicted heart disease risk along with an estimated probability.

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Machine learning can help identify patterns in medical data that may be associated with heart disease.

This project builds an end-to-end machine learning pipeline that:

- Loads and cleans heart disease data
- Handles missing values
- Performs exploratory data analysis (EDA)
- Engineers and categorizes features
- Splits data into training and testing sets
- Applies preprocessing using Scikit-learn pipelines
- Trains multiple machine learning models
- Evaluates models using multiple classification metrics
- Performs hyperparameter tuning
- Uses stratified cross-validation
- Selects the best-performing model
- Calculates feature importance
- Saves the trained model and preprocessing pipeline
- Provides predictions through a Streamlit web application

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a binary heart disease classification model.
2. Compare different machine learning algorithms.
3. Identify the best-performing model.
4. Improve model performance using hyperparameter tuning.
5. Validate model performance using cross-validation.
6. Understand which features contribute most to predictions.
7. Deploy the model as an interactive web application.

---

## 📊 Dataset

The project uses the **UCI Heart Disease – Cleveland dataset**.

### Dataset Information

- **Samples:** 303
- **Original Features:** 13
- **Target:** Heart disease presence
- **Problem Type:** Binary Classification

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `target` | Heart disease diagnosis |

### Target Transformation

The original dataset contains five target values:

```text
0 → No Heart Disease
1 → Heart Disease
2 → Heart Disease
3 → Heart Disease
4 → Heart Disease