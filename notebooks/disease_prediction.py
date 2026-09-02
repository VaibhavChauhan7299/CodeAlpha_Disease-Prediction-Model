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
