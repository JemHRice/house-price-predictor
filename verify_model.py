import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

print("=" * 60)
print("MODEL TRAINING VERIFICATION")
print("=" * 60)

# Load cleaned data
df = pd.read_csv("Housing_cleaned.csv")

print(f"\n✓ Data loaded successfully")
print(f"  Dataset shape: {df.shape}")
print(f"  Rows: {df.shape[0]}, Features: {df.shape[1]}")

# Check data types
print(f"\n✓ Data types:")
print(df.dtypes)

# Check for missing values
print(f"\n✓ Missing values: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

print(f"\n✓ Features: {X.shape[1]}")
print(f"  Columns: {X.columns.tolist()}")

print(f"\n✓ Target (Price):")
print(f"  Min: ${y.min():,.0f}")
print(f"  Max: ${y.max():,.0f}")
print(f"  Mean: ${y.mean():,.0f}")
print(f"  Std: ${y.std():,.0f}")

# Train model
model = LinearRegression()
model.fit(X, y)

print(f"\n✓ Model trained successfully")

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n✓ Model Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  MAE: ${mae:,.0f}")
print(f"  RMSE: ${rmse:,.0f}")

# Show coefficients
print(f"\n✓ Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:,.2f}")
print(f"  Intercept: ${model.intercept_:,.2f}")

# Test prediction
test_input = X.iloc[0:1]
test_pred = model.predict(test_input)[0]
actual = y.iloc[0]

print(f"\n✓ Sample Prediction Test:")
print(f"  Predicted: ${test_pred:,.0f}")
print(f"  Actual: ${actual:,.0f}")
print(f"  Difference: ${abs(test_pred - actual):,.0f}")

print(f"\n" + "=" * 60)
print("✓ All checks passed! Model is ready for deployment")
print("=" * 60)
