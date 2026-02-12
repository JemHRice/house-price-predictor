"""
Advanced house price prediction model for Sydney
Combines three strategies to reduce prediction error:
1. Suburb clustering by price tier
2. Comprehensive feature engineering
3. Price-per-sqm prediction model
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


def train_advanced_model(
    data_file="domain_properties_cleaned.csv", model_file="model.pkl"
):
    """
    Train advanced price prediction model with three improvements:
    1. Suburb clustering - group similar suburbs together
    2. Feature engineering - add suburb statistics and temporal features
    3. Price-per-sqm prediction - normalize the huge variance in absolute prices
    """

    print("=" * 80)
    print("TRAINING ADVANCED PRICE PREDICTION MODEL")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} records")

    target_column = "price_adjusted_to_2026"

    # ============================================================
    # STEP 1: SUBURB CLUSTERING
    # ============================================================
    print("\n2. Creating suburb clusters by price tier...")

    # Calculate suburb statistics
    suburb_stats = (
        df.groupby("suburb")
        .agg(
            {
                target_column: ["median", "mean", "std", "count"],
                "property_size_sqm": "mean",
            }
        )
        .reset_index()
    )
    suburb_stats.columns = [
        "suburb",
        "median_price",
        "mean_price",
        "price_std",
        "count",
        "avg_size",
    ]

    # Cluster suburbs by median price (K-means on median price)
    X_cluster = suburb_stats[["median_price"]].values
    kmeans = KMeans(n_clusters=5, random_state=42)
    suburb_stats["price_cluster"] = kmeans.fit_predict(X_cluster)

    # Create mappings
    suburb_to_cluster = dict(zip(suburb_stats["suburb"], suburb_stats["price_cluster"]))
    suburb_to_median = dict(zip(suburb_stats["suburb"], suburb_stats["median_price"]))

    print(f"   Created 5 price clusters:")
    for cluster_id in range(5):
        cluster_data = suburb_stats[suburb_stats["price_cluster"] == cluster_id]
        min_price = cluster_data["median_price"].min()
        max_price = cluster_data["median_price"].max()
        count = len(cluster_data)
        print(
            f"     Cluster {cluster_id}: {count} suburbs, median price ${min_price:,.0f} - ${max_price:,.0f}"
        )

    # ============================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================
    print("\n3. Feature engineering...")

    # Add suburb-level features
    df = df.merge(
        suburb_stats[
            ["suburb", "price_cluster", "median_price", "price_std", "avg_size"]
        ],
        on="suburb",
        how="left",
    )

    # Feature 1: Price per sqm (normalize variance)
    df["price_per_sqm"] = df[target_column] / df["property_size_sqm"]

    # Feature 2: Deviation from suburb median (is this property over/under-valued for its suburb?)
    df["price_deviation_from_median"] = df[target_column] - df["median_price"]

    # Feature 3: Price per sqm ratio to suburb average
    df["size_ratio_to_suburb"] = df["property_size_sqm"] / df["avg_size"]

    # Feature 4: Price volatility in suburb (higher std = more unpredictable)
    df["suburb_price_volatility"] = df["price_std"] / df["median_price"]

    # Feature 5: Extract temporal features
    df["date_sold"] = pd.to_datetime(df["date_sold"], format="%m/%Y")
    df["year_sold"] = df["date_sold"].dt.year
    df["month_sold"] = df["date_sold"].dt.month
    df["quarter_sold"] = df["date_sold"].dt.quarter

    # Feature 6: Beds + baths combined
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]

    # Feature 7: Parking to size ratio
    df["parking_per_sqm"] = df["parking"] / (df["property_size_sqm"] / 100)

    print(f"   ✓ Created 7 new engineered features")

    # ============================================================
    # STEP 3: PREPARE DATA FOR MODELING
    # ============================================================
    print("\n4. Preparing features for modeling...")

    feature_columns = [
        "suburb",
        "bedrooms",
        "bathrooms",
        "parking",
        "property_size_sqm",
        "type",
        "property_inflation_index",
        "price_cluster",
        "price_deviation_from_median",
        "size_ratio_to_suburb",
        "suburb_price_volatility",
        "year_sold",
        "quarter_sold",
        "rooms_total",
        "parking_per_sqm",
    ]

    # Target: price_per_sqm (normalized, lower variance)
    TARGET = "price_per_sqm"

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    print(f"   Features ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"   Target: {TARGET}")
    print(f"   Target range (price/sqm): ${y.min():,.0f} - ${y.max():,.0f}/sqm")
    print(f"   Target mean: ${y.mean():,.0f}/sqm")
    print(f"   Target std: ${y.std():,.0f}/sqm (much lower than absolute prices!)")

    # Encode categorical variables
    print("\n5. Encoding categorical features...")
    label_encoders = {}
    for col in ["suburb", "type"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} categories")

    # Train-test split
    print("\n6. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} records, Test: {len(X_test)} records")

    # Scale features
    print("\n7. Scaling numerical features...")
    scaler = StandardScaler()
    numerical_cols = [col for col in feature_columns if col not in ["suburb", "type"]]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print(f"   Scaled {len(numerical_cols)} numerical features")

    # ============================================================
    # STEP 4: TRAIN PRICE-PER-SQM MODEL
    # ============================================================
    print("\n8. Training XGBoost model on price-per-sqm...")

    xgb_params = {
        "n_estimators": 600,
        "max_depth": 10,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 3,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False
    )
    print("   ✓ Model trained!")

    # Feature importance
    print("\n9. Feature importance:")
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # ============================================================
    # STEP 5: EVALUATE MODEL
    # ============================================================
    print("\n10. Evaluating model...")

    # Predictions in price-per-sqm
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Convert back to absolute price for MAE comparison
    X_test_copy = X_test.copy()
    df_test = X_test_copy.join(y_test.reset_index(drop=True))
    df_test["price_per_sqm_actual"] = y_test.values
    df_test["price_per_sqm_pred"] = y_test_pred
    df_test["size"] = df.loc[X_test.index, "property_size_sqm"].values
    df_test["price_pred_absolute"] = df_test["price_per_sqm_pred"] * df_test["size"]
    df_test["price_actual_absolute"] = df.loc[X_test.index, target_column].values

    # Metrics
    mae_per_sqm = mean_absolute_error(y_test, y_test_pred)
    r2_per_sqm = r2_score(y_test, y_test_pred)
    mae_absolute = mean_absolute_error(
        df_test["price_actual_absolute"], df_test["price_pred_absolute"]
    )

    print(f"   Price-per-sqm MAE: ${mae_per_sqm:,.0f}/sqm")
    print(f"   Price-per-sqm R²: {r2_per_sqm:.4f}")
    print(f"   Absolute price MAE: ${mae_absolute:,.0f}")
    print(
        f"   Absolute price R²: {r2_score(df_test['price_actual_absolute'], df_test['price_pred_absolute']):.4f}"
    )

    # Calculate percentage error
    mape = (
        np.mean(
            np.abs(
                (df_test["price_actual_absolute"] - df_test["price_pred_absolute"])
                / df_test["price_actual_absolute"]
            )
        )
        * 100
    )
    print(f"   Mean Absolute Percentage Error: {mape:.1f}%")

    # ============================================================
    # STEP 6: SAVE MODEL
    # ============================================================
    print(f"\n11. Saving model package...")

    model_package = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
        "numerical_cols": numerical_cols,
        "target_type": "price_per_sqm",  # Important: we predict per sqm!
        "suburb_to_cluster": suburb_to_cluster,
        "suburb_to_median": suburb_to_median,
        "suburb_stats": suburb_stats.to_dict("list"),
        "kmeans": kmeans,
        "training_config": {
            "mae_per_sqm": float(mae_per_sqm),
            "mae_absolute": float(mae_absolute),
            "r2_per_sqm": float(r2_per_sqm),
            "mape": float(mape),
            "test_samples": len(X_test),
            "train_samples": len(X_train),
            "algorithm": "XGBoost (Advanced)",
            "improvements": [
                "Suburb clustering",
                "Feature engineering",
                "Price-per-sqm normalization",
            ],
        },
    }

    with open(model_file, "wb") as f:
        pickle.dump(model_package, f)
    print(f"   ✓ Saved to {model_file}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - ADVANCED MODEL")
    print("=" * 80)
    print(f"\n✓ Model improvements achieved:")
    print(f"  • Suburb clustering: 5 price tiers for better accuracy")
    print(f"  • Feature engineering: 7 new features for better patterns")
    print(f"  • Price-per-sqm normalization: Reduced variance dramatically")
    print(f"\n✓ Performance:")
    print(f"  • Absolute MAE: ${mae_absolute:,.0f} (vs ${444698:,.0f} before)")
    print(f"  • MAPE: {mape:.1f}% (percentage error)")
    print(f"  • R² Score: {r2_per_sqm:.4f}")

    return model_package


if __name__ == "__main__":
    train_advanced_model()
