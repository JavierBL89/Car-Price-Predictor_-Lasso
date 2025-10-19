# ======================================================
#  FINAL MODEL: Ridge Regression on Cleaned Dataset
# ======================================================

# --- Import libraries ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1️⃣ Separate features and target variable ---
# 'price' is the dependent variable (target)
# All other columns are independent variables (features)
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

# --- 2️⃣ Standardize features ---
# Ridge regression is sensitive to feature scale.
# StandardScaler ensures all features are on the same scale (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3️⃣ Initialize and train Ridge Regression model ---
# alpha controls regularization strength:
#   - small alpha = weaker penalty (acts like OLS)
#   - large alpha = stronger penalty (reduces coefficient size)
ridge = Ridge(alpha=10, random_state=42)
ridge.fit(X_scaled, y)

# --- 4️⃣ Make predictions on the training data ---
# (Later you can replace this with test data if you split your dataset)
y_pred = ridge.predict(X_scaled)

# --- 5️⃣ Evaluate model performance ---
# R² (coefficient of determination): proportion of variance explained by model
# RMSE (root mean squared error): average prediction error in same units as price
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"✅ R² Score: {r2:.3f}")
print(f"✅ RMSE: {rmse:.2f}")

# --- 6️⃣ Analyze feature importance ---
# Ridge coefficients represent how each feature influences price.
# Positive → increases price; Negative → decreases price
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n📊 Top 10 Influential Features:")
print(importance.head(10))

# --- 7️⃣ (Optional) Visualize top 10 coefficients ---
plt.figure(figsize=(8,5))
plt.barh(importance["Feature"].head(10), importance["Coefficient"].head(10))
plt.gca().invert_yaxis()
plt.title("Top 10 Positive Ridge Coefficients")
plt.xlabel("Coefficient Value")
plt.show()
