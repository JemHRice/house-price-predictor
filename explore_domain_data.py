"""
Data Exploration for Domain Properties Dataset
Analyzes Sydney house price data from 2016-2021 to understand structure and quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("domain_properties.csv")

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn Names and Types:")
print(df.dtypes)

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame(
    {
        "Column": missing.index,
        "Missing_Count": missing.values,
        "Percentage": missing_pct.values,
    }
).sort_values("Missing_Count", ascending=False)
print(missing_df[missing_df["Missing_Count"] > 0])
if missing_df["Missing_Count"].sum() == 0:
    print("No missing values found!")

print("\n" + "=" * 80)
print("BASIC STATISTICS - NUMERICAL COLUMNS")
print("=" * 80)
print(df.describe().T)

print("\n" + "=" * 80)
print("CATEGORICAL COLUMNS ANALYSIS")
print("=" * 80)
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Top 5 values:")
    print(df[col].value_counts().head())

print("\n" + "=" * 80)
print("PRICE ANALYSIS (TARGET VARIABLE)")
print("=" * 80)
print(f"Price Statistics:")
print(f"  Min: ${df['price'].min():,.0f}")
print(f"  Max: ${df['price'].max():,.0f}")
print(f"  Mean: ${df['price'].mean():,.0f}")
print(f"  Median: ${df['price'].median():,.0f}")
print(f"  Std Dev: ${df['price'].std():,.0f}")

# Check for outliers
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["price"] < Q1 - 1.5 * IQR) | (df["price"] > Q3 + 1.5 * IQR)]
print(
    f"\nPotential outliers (IQR method): {len(outliers)} records ({len(outliers)/len(df)*100:.2f}%)"
)

print("\n" + "=" * 80)
print("DATE ANALYSIS")
print("=" * 80)
df["date_sold"] = pd.to_datetime(df["date_sold"], format="%d/%m/%y")
print(f"Date Range: {df['date_sold'].min()} to {df['date_sold'].max()}")
print(f"\nSales by Year:")
print(df["date_sold"].dt.year.value_counts().sort_index())
print(f"\nSales by Month (first year):")
print(df["date_sold"].dt.month.value_counts().sort_index())

print("\n" + "=" * 80)
print("PROPERTY TYPE ANALYSIS")
print("=" * 80)
print(df["type"].value_counts())

print("\n" + "=" * 80)
print("BEDROOM & BATHROOM ANALYSIS")
print("=" * 80)
print(f"\nBedrooms distribution:")
print(df["num_bed"].value_counts().sort_index())
print(f"\nBathrooms distribution:")
print(df["num_bath"].value_counts().sort_index())

print("\n" + "=" * 80)
print("PROPERTY SIZE ANALYSIS")
print("=" * 80)
print(f"Property Size (sqm) Statistics:")
print(f"  Min: {df['property_size'].min():,.0f}")
print(f"  Max: {df['property_size'].max():,.0f}")
print(f"  Mean: {df['property_size'].mean():,.2f}")
print(f"  Median: {df['property_size'].median():,.2f}")

print("\n" + "=" * 80)
print("SUBURB ANALYSIS")
print("=" * 80)
print(f"Total unique suburbs: {df['suburb'].nunique()}")
print(f"\nTop 10 suburbs by number of sales:")
print(df["suburb"].value_counts().head(10))
print(f"\nTop 10 suburbs by median price:")
suburb_prices = df.groupby("suburb")["price"].median().sort_values(ascending=False)
print(suburb_prices.head(10))

print("\n" + "=" * 80)
print("SAMPLE DATA")
print("=" * 80)
print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

# Check for suspicious zero values
zero_checks = {
    "num_bed == 0": len(df[df["num_bed"] == 0]),
    "num_bath == 0": len(df[df["num_bath"] == 0]),
    "num_parking == 0": len(df[df["num_parking"] == 0]),
    "property_size == 0": len(df[df["property_size"] == 0]),
    "price == 0": len(df[df["price"] == 0]),
}
print("\nRecords with zero/suspicious values:")
for check, count in zero_checks.items():
    print(f"  {check}: {count}")

# Correlation analysis for numerical columns
print("\n" + "=" * 80)
print("CORRELATION WITH PRICE")
print("=" * 80)
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numerical_cols].corr()["price"].sort_values(ascending=False)
print(correlations)

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Review data quality issues and decide on cleaning strategy")
print("2. Check if vacant land and houses should be analyzed separately")
print("3. Decide on handling zero values in numeric fields")
print("4. Consider feature engineering opportunities (e.g., suburb clustering)")
print("5. Analyze price by property type and location")
