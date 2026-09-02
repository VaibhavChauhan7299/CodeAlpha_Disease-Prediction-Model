import pandas as pd
import numpy as np

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