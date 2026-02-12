"""
Data Cleaning for Domain Properties Dataset
Prepares Sydney house price data for model training
"""

import pandas as pd
import numpy as np


def clean_domain_data(
    input_file="domain_properties.csv", output_file="domain_properties_cleaned.csv"
):
    """
    Clean the domain properties dataset

    Steps:
    1. Load the data
    2. Convert date column to datetime
    3. Handle missing values
    4. Remove/filter outliers
    5. Handle suspicious zero values
    6. Filter property types if needed
    7. Validate and save cleaned data
    """

    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Convert date column to month_year format
    print("\n1. Converting date_sold to month_year format...")
    df["date_sold_dt"] = pd.to_datetime(df["date_sold"], format="%d/%m/%y")
    df["year_sold"] = df["date_sold_dt"].dt.year
    df["date_sold"] = df["date_sold_dt"].dt.strftime("%m/%Y")
    print(f"   Date column converted to month/year format")

    # Apply inflation adjustment to 2026 prices
    print("\n2. Applying inflation adjustment to 2026...")
    print("   Inflation assumptions:")
    print("   - 2016-2025: 4.5% annual inflation (RBA inflation calculator)")
    print("   - 2025-2026: 4.0% predicted inflation")

    def calculate_inflation_multiplier(year):
        """Calculate inflation multiplier from sale year to 2026"""
        if year >= 2026:
            return 1.0
        elif year == 2025:
            return 1.04
        else:
            years_at_4_5 = 2025 - year
            multiplier = (1.045**years_at_4_5) * 1.04
            return multiplier

    df["inflation_multiplier"] = df["year_sold"].apply(calculate_inflation_multiplier)
    df["price_adjusted_to_2026"] = (df["price"] * df["inflation_multiplier"]).round(2)
    print(f"   Prices adjusted to 2026 equivalent values")

    # Handle missing values
    print("\n3. Handling missing values...")
    missing_before = df.isnull().sum().sum()

    # Drop rows with missing values (if any)
    df = df.dropna()

    missing_after = df.isnull().sum().sum()
    print(f"   Missing values removed: {missing_before - missing_after}")
    print(f"   Remaining rows: {df.shape[0]}")

    # Remove outliers using IQR method on price
    print("\n4. Removing price outliers (IQR method)...")
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_before = len(df)
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
    outliers_removed = outliers_before - len(df)
    print(f"   Price bounds: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    print(
        f"   Outliers removed: {outliers_removed} ({outliers_removed/outliers_before*100:.2f}%)"
    )

    # Filter to only houses (exclude vacant land)
    print("\n5. Filtering property types...")
    print(f"   Property types before: {df['type'].value_counts().to_dict()}")
    df = df[df["type"] != "Vacant land"]
    print(f"   Property types after: {df['type'].value_counts().to_dict()}")
    print(f"   Rows remaining: {len(df)}")

    # Remove or flag records with missing bedroom/bathroom values (0 might mean missing)
    print("\n6. Handling zero values in essential features...")
    zero_bed_bath = len(df[(df["num_bed"] == 0) | (df["num_bath"] == 0)])
    print(f"   Records with 0 beds or baths: {zero_bed_bath}")

    if zero_bed_bath > 0:
        print(f"   Removing records where bed==0 or bath==0...")
        df = df[(df["num_bed"] > 0) & (df["num_bath"] > 0)]
        print(f"   Rows remaining: {len(df)}")

    # Verify property size is valid
    print("\n7. Validating property size...")
    zero_size = len(df[df["property_size"] == 0])
    if zero_size > 0:
        print(f"   Found {zero_size} records with property_size == 0")
        df = df[df["property_size"] > 0]
        print(f"   Rows remaining: {len(df)}")

    # Data type optimization
    print("\n8. Optimizing data types...")
    df["num_bed"] = df["num_bed"].astype("int8")
    df["num_bath"] = df["num_bath"].astype("int8")
    df["num_parking"] = df["num_parking"].astype("int8")
    df["cash_rate"] = df["cash_rate"].astype("int8")

    # Remove unwanted columns
    print("\n9. Removing unwanted columns...")
    columns_to_drop = [
        "km_from_cbd",
        "suburb_elevation",
        "suburb_lng",
        "suburb_lat",
        "suburb_population",
        "suburb_sqkm",
        "suburb_median_income",
        "cash_rate",
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    print(f"   Removed columns: {columns_to_drop}")

    # Rename columns for clarity
    print("\n10. Renaming columns for consistency...")
    column_mapping = {
        "num_bed": "bedrooms",
        "num_bath": "bathrooms",
        "num_parking": "parking",
        "property_size": "property_size_sqm",
        "suburb_sqkm": "suburb_area_sqkm",
    }
    df = df.rename(columns=column_mapping)

    # Remove temporary columns used for calculations
    print("\n11. Removing temporary calculation columns...")
    temp_cols_to_drop = ["date_sold_dt", "year_sold", "inflation_multiplier"]
    df = df.drop(columns=temp_cols_to_drop, errors="ignore")
    print(f"   Removed temporary columns: {temp_cols_to_drop}")

    # Final validation
    print("\n12. Final validation...")
    print(f"   Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(
        f"   Price range (original): ${df['price'].min():,.0f} - ${df['price'].max():,.0f}"
    )
    print(
        f"   Price range (2026 adjusted): ${df['price_adjusted_to_2026'].min():,.0f} - ${df['price_adjusted_to_2026'].max():,.0f}"
    )
    print(f"   Bedrooms range: {df['bedrooms'].min()} - {df['bedrooms'].max()}")
    print(f"   Bathrooms range: {df['bathrooms'].min()} - {df['bathrooms'].max()}")
    print(
        f"   Property size range: {df['property_size_sqm'].min():,.0f} - {df['property_size_sqm'].max():,.0f} sqm"
    )
    print(f"   Columns: {df.columns.tolist()}")

    # Save cleaned data
    print(f"\n13. Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"    ✓ Cleaned data saved successfully!")

    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Original rows: {pd.read_csv(input_file).shape[0]}")
    print(f"Cleaned rows: {df.shape[0]}")
    print(f"Rows removed: {pd.read_csv(input_file).shape[0] - df.shape[0]}")
    print(
        f"Retention rate: {df.shape[0] / pd.read_csv(input_file).shape[0] * 100:.2f}%"
    )

    return df


if __name__ == "__main__":
    df_clean = clean_domain_data()
    print("\nCleaning complete! Data is ready for modeling.")
    print("\nFirst few rows of cleaned data:")
    print(df_clean.head())
