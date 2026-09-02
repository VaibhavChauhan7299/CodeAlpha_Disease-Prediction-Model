import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/heart_disease.csv",
    header=None
)

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display data types
print("\nData types:")
print(df.dtypes)

print("\nMissing value representation:")
print((df == "?").sum())