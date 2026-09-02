import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(
    "data/heart_disease.csv",
    header=None
)

# Rename columns
df.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display data types
print("\nData types:")
print(df.dtypes)

# Convert ? to NaN
df = df.replace("?", np.nan)

# Convert all columns to numeric
df = df.astype(float)

print("\nData types after conversion:")
print(df.dtypes)

# Display missing values
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check target values
print("\nTarget values:")
print(df["target"].value_counts().sort_index())

# Check for invalid/missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].median())

# Convert target to binary classification
# 0 = No heart disease
# 1-4 = Heart disease
df["target"] = (df["target"] > 0).astype(int)

# Final missing value check
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Final target distribution
print("\nTarget distribution:")
print(df["target"].value_counts())

# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import os

# Create folder for EDA graphs
os.makedirs("eda_plots", exist_ok=True)

print("\n========== EDA ==========")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("eda_plots/target_distribution.png")
plt.close()


# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("eda_plots/age_distribution.png")
plt.close()


# 3. Age vs Heart Disease
plt.figure(figsize=(8, 5))
sns.boxplot(x="target", y="age", data=df)
plt.title("Age vs Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("eda_plots/age_vs_disease.png")
plt.close()


# 4. Sex vs Heart Disease
plt.figure(figsize=(6, 4))
sns.countplot(x="sex", hue="target", data=df)
plt.title("Heart Disease by Sex")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")
plt.legend(title="Heart Disease")
plt.tight_layout()
plt.savefig("eda_plots/sex_vs_disease.png")
plt.close()


# 5. Chest Pain vs Heart Disease
plt.figure(figsize=(8, 5))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Heart Disease vs Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.legend(title="Heart Disease")
plt.tight_layout()
plt.savefig("eda_plots/chest_pain_vs_disease.png")
plt.close()


# 6. Cholesterol Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="chol", bins=20, kde=True)
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("eda_plots/cholesterol_distribution.png")
plt.close()


# 7. Maximum Heart Rate Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="thalach", bins=20, kde=True)
plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("eda_plots/heart_rate_distribution.png")
plt.close()


# 8. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("EDA plots saved in: eda_plots/")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\n========== FEATURE ENGINEERING ==========")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

# Categorical features
categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

# Numerical features
numerical_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

print("\nNumerical feature summary:")
print(X[numerical_features].describe())

print("\nCategorical feature values:")

for column in categorical_features:
    print(f"\n{column}:")
    print(X[column].value_counts().sort_index())

print("\nFeature engineering completed successfully!")

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

print("\n========== TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

print("\nTrain/Test split completed successfully!")

# ==========================================
# DATA PREPROCESSING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("\n========== DATA PREPROCESSING ==========")

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nOriginal training shape:")
print(X_train.shape)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nOriginal testing shape:")
print(X_test.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

print("\nData preprocessing completed successfully!")


# ==========================================
# LOGISTIC REGRESSION
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n========== LOGISTIC REGRESSION ==========")

# Create model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
logistic_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_lr = logistic_model.predict(X_test_processed)

# Evaluate model
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print("\nLogistic Regression Results:")
print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1 Score:  {lr_f1:.4f}")

# ==========================================
# DECISION TREE
# ==========================================

from sklearn.tree import DecisionTreeClassifier

print("\n========== DECISION TREE ==========")

# Create model
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

# Train model
decision_tree_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_dt = decision_tree_model.predict(X_test_processed)

# Evaluate model
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print("\nDecision Tree Results:")
print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1 Score:  {dt_f1:.4f}")

# ==========================================
# RANDOM FOREST
# ==========================================

from sklearn.ensemble import RandomForestClassifier

print("\n========== RANDOM FOREST ==========")

# Create model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
random_forest_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_rf = random_forest_model.predict(X_test_processed)

# Evaluate model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")

# ==========================================
# SUPPORT VECTOR MACHINE (SVM)
# ==========================================

from sklearn.svm import SVC

print("\n========== SUPPORT VECTOR MACHINE ==========")

# Create model
svm_model = SVC(
    probability=True,
    random_state=42
)

# Train model
svm_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_processed)

# Evaluate model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print("\nSVM Results:")
print(f"Accuracy:  {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall:    {svm_recall:.4f}")
print(f"F1 Score:  {svm_f1:.4f}")

# ==========================================
# XGBOOST
# ==========================================

from xgboost import XGBClassifier

print("\n========== XGBOOST ==========")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# Train the model
xgb_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test_processed)

# Calculate metrics
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1 Score:  {xgb_f1:.4f}")

# ==========================================
# MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("\n========== MODEL EVALUATION ==========")

# Store predictions
predictions = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "XGBoost": y_pred_xgb
}

# Store probability predictions
probabilities = {
    "Logistic Regression": logistic_model.predict_proba(X_test_processed)[:, 1],
    "Decision Tree": decision_tree_model.predict_proba(X_test_processed)[:, 1], # type: ignore
    "Random Forest": random_forest_model.predict_proba(X_test_processed)[:, 1],
    "SVM": svm_model.predict_proba(X_test_processed)[:, 1],
    "XGBoost": xgb_model.predict_proba(X_test_processed)[:, 1]
}

# Evaluate each model
evaluation_results = []

for model_name in predictions:

    y_pred = predictions[model_name]
    y_prob = probabilities[model_name]

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

# Create comparison DataFrame
evaluation_df = pd.DataFrame(evaluation_results)

print("\nModel Evaluation Results:")
print(evaluation_df.round(4).to_string(index=False))

# ==========================================
# CONFUSION MATRIX
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== CONFUSION MATRICES ==========")

for model_name, y_pred in predictions.items():

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        f"eda_plots/{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

print("\nConfusion matrices saved successfully!")

# ==========================================
# ROC CURVE
# ==========================================

from sklearn.metrics import roc_curve

print("\n========== ROC CURVE ==========")

plt.figure(figsize=(8, 6))

for model_name, y_prob in probabilities.items():

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc_score = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_score:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "eda_plots/roc_curve_comparison.png",
    dpi=300
)

plt.close()

print("\nROC curve saved successfully!")

# ==========================================
# HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV

print("\n========== HYPERPARAMETER TUNING ==========")

# ------------------------------------------
# Logistic Regression
# ------------------------------------------

lr_params = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

lr_grid.fit(X_train_processed, y_train)

print("\nBest Logistic Regression Parameters:")
print(lr_grid.best_params_)

print("Best Logistic Regression CV ROC-AUC:")
print(f"{lr_grid.best_score_:.4f}")


# ------------------------------------------
# Random Forest
# ------------------------------------------

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train_processed, y_train)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

print("Best Random Forest CV ROC-AUC:")
print(f"{rf_grid.best_score_:.4f}")


# ------------------------------------------
# SVM
# ------------------------------------------

svm_params = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

svm_grid.fit(X_train_processed, y_train)

print("\nBest SVM Parameters:")
print(svm_grid.best_params_)

print("Best SVM CV ROC-AUC:")
print(f"{svm_grid.best_score_:.4f}")


# ------------------------------------------
# XGBoost
# ------------------------------------------

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ),
    xgb_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

xgb_grid.fit(X_train_processed, y_train)

print("\nBest XGBoost Parameters:")
print(xgb_grid.best_params_)

print("Best XGBoost CV ROC-AUC:")
print(f"{xgb_grid.best_score_:.4f}")


print("\nHyperparameter tuning completed successfully!")

# ==========================================
# CROSS VALIDATION & FINAL MODEL SELECTION
# ==========================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("\n========== CROSS VALIDATION ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Best tuned models
best_lr_model = lr_grid.best_estimator_
best_rf_model = rf_grid.best_estimator_
best_svm_model = svm_grid.best_estimator_
best_xgb_model = xgb_grid.best_estimator_

tuned_models = {
    "Logistic Regression": best_lr_model,
    "Random Forest": best_rf_model,
    "SVM": best_svm_model,
    "XGBoost": best_xgb_model
}

cv_results = []

for model_name, model in tuned_models.items():

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Mean ROC-AUC": scores.mean(),
        "Std ROC-AUC": scores.std()
    })

    print(f"\n{model_name}:")
    print(f"Fold ROC-AUC: {scores}")
    print(f"Mean ROC-AUC: {scores.mean():.4f}")
    print(f"Std ROC-AUC:  {scores.std():.4f}")


# Create comparison DataFrame
cv_results_df = pd.DataFrame(cv_results)

print("\n========== CROSS-VALIDATION RESULTS ==========")
print(cv_results_df.round(4).to_string(index=False))


# Select best model based on mean ROC-AUC
best_model_name = cv_results_df.loc[
    cv_results_df["Mean ROC-AUC"].idxmax(),
    "Model"
]

best_model = tuned_models[best_model_name] # pyright: ignore[reportArgumentType]

print("\n========== FINAL MODEL ==========")
print(f"Best Model: {best_model_name}")

print("\nCross-validation results:")
print(cv_results_df.round(4).to_string(index=False))

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

from sklearn.inspection import permutation_importance

print("\n========== FEATURE IMPORTANCE ==========")

# Calculate permutation importance
importance = permutation_importance(
    best_model,
    X_test_processed,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get processed feature names
feature_names = preprocessor.get_feature_names_out()

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance.importances_mean # pyright: ignore[reportAttributeAccessIssue]
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(
    feature_importance_df.head(15).round(4).to_string(index=False)
)

# Save feature importance
feature_importance_df.to_csv(
    "eda_plots/feature_importance.csv",
    index=False
)

# Plot top 15 features
plt.figure(figsize=(10, 6))

top_features = feature_importance_df.head(15).sort_values(
    by="Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance - SVM")

plt.tight_layout()

plt.savefig(
    "eda_plots/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFeature importance plot saved successfully!")

# ==========================================
# SAVE BEST MODEL AND PREPROCESSOR
# ==========================================

import joblib
import os

print("\n========== SAVING MODEL ==========")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save best model
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

# Save preprocessor
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/best_model.pkl")

print("\nPreprocessor saved successfully!")
print("Location: models/preprocessor.pkl")

# ==========================================
# LOAD & VERIFY SAVED MODEL
# ==========================================

import joblib

print("\n========== LOADING SAVED MODEL ==========")

# Load model
loaded_model = joblib.load("models/best_model.pkl")

# Load preprocessor
loaded_preprocessor = joblib.load("models/preprocessor.pkl")

print("\nSaved model loaded successfully!")
print("Saved preprocessor loaded successfully!")

# Transform test data using loaded preprocessor
X_test_loaded = loaded_preprocessor.transform(X_test)

# Make predictions using loaded model
y_pred_loaded = loaded_model.predict(X_test_loaded)

# Check predictions
print("\nFirst 10 predictions:")
print(y_pred_loaded[:10])

print("\nFirst 10 actual values:")
print(y_test.values[:10])

# Check accuracy
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)

print(f"\nLoaded model accuracy: {loaded_accuracy:.4f}")

print("\nModel verification completed successfully!")

# ==========================================
# NEW PATIENT PREDICTION
# ==========================================

print("\n========== NEW PATIENT PREDICTION ==========")

# Create a new patient
new_patient = pd.DataFrame({
    "age": [55],
    "sex": [1],
    "cp": [4],
    "trestbps": [140],
    "chol": [250],
    "fbs": [0],
    "restecg": [1],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [1.2],
    "slope": [2],
    "ca": [0],
    "thal": [3]
})

# Transform patient data
new_patient_processed = loaded_preprocessor.transform(new_patient)

# Make prediction
prediction = loaded_model.predict(new_patient_processed)[0]

# Get probability
probability = loaded_model.predict_proba(
    new_patient_processed
)[0][1]

# Display result
if prediction == 1:
    print("\nPrediction: Heart Disease Risk")
else:
    print("\nPrediction: No Heart Disease Risk")

print(f"Probability: {probability:.2%}")