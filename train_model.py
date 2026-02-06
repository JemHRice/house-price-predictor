import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

print("=" * 60)
print("TRAINING LINEAR REGRESSION MODEL")
print("=" * 60)

# Load cleaned data
df = pd.read_csv("Housing_cleaned.csv")

print(f"\n Data loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Separate features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n Train/Test Split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Train Linear Regression model
print(f"\n Training Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics on test set (more realistic)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Also show training metrics for comparison
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

print(f"\n Training Performance:")
print(f"  R Score: {r2_train:.4f}")
print(f"  MAE: $" "{mae_train:,.0f}")

print(f"\n Test Performance (Final Model):")
print(f"  R Score: {r2_test:.4f}")
print(f"  MAE: $" "{mae_test:,.0f}")
print(f"  RMSE: $" "{rmse_test:,.0f}")

# Show feature coefficients
print(f"\n Feature Coefficients:")
feature_coefs = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_}
).sort_values("Coefficient", ascending=False, key=abs)

for idx, row in feature_coefs.iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:,.2f}")
print(f"  Intercept: $" "{model.intercept_:,.2f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\n Model saved as model.pkl")

print(f"\n" + "=" * 60)
print(" Training Complete!")
print("=" * 60)
