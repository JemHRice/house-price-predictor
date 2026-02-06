import pandas as pd
import numpy as np

# Load the raw CSV
df = pd.read_csv("Housing.csv")

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())

# Select only the columns we need
columns_to_keep = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]
df = df[columns_to_keep]

print("\nDataset shape after column selection:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# No missing values - can continue

# One-hot encode furnishingstatus
print("\nUnique furnishing statuses:", df["furnishingstatus"].unique())
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=False)

# Convert yes/no columns to 1/0
df["airconditioning"] = (df["airconditioning"] == "yes").astype(int)
df["prefarea"] = (df["prefarea"] == "yes").astype(int)

# Rename columns
df = df.rename(
    columns={
        "price": "Price",
        "area": "Area",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "airconditioning": "Air Conditioning",
        "parking": "Parking",
        "prefarea": "Preferred Area",
        "furnishingstatus_furnished": "Fully Furnished",
        "furnishingstatus_semi-furnished": "Semi Furnished",
        "furnishingstatus_unfurnished": "Unfurnished",
    }
)

# Convert all columns to numeric to ensure compatibility with sklearn
df = df.astype("float64")

print("\nFinal columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print("\nFinal columns after encoding:")
print(df.columns.tolist())

print("\nDataset shape after encoding:", df.shape)

# Save cleaned data
df.to_csv("Housing_cleaned.csv", index=False)
print("\n✓ Cleaned data saved to Housing_cleaned.csv")
