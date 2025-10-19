# ======================================================
#  LASSO REGRESSION MODEL (Feature Selection)
# ======================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1️⃣ Separate features and target variable ---
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

# --- 2️⃣ Standardize features ---
# Lasso (like Ridge) is sensitive to feature scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3️⃣ Initialize and train Lasso model ---
# alpha controls the regularization strength
# Higher alpha → more coefficients shrink to 0 (stronger feature selection)
lasso = Lasso(alpha=100, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# --- 4️⃣ Predict and evaluate model ---
y_pred = lasso.predict(X_scaled)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"✅ R² Score: {r2:.3f}")
print(f"✅ RMSE: {rmse:.2f}")

# --- 5️⃣ Check coefficients (feature importance) ---
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lasso.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n📊 Top 10 Influential Features:")
print(importance.head(10))

# --- 6️⃣ Count how many features were dropped (coeff = 0) ---
zero_coef = (importance["Coefficient"] == 0).sum()
print(f"\n🚫 Features eliminated by Lasso: {zero_coef} / {len(importance)}")

# --- 7️⃣ Visualize top coefficients ---
plt.figure(figsize=(8,5))
plt.barh(importance["Feature"].head(10), importance["Coefficient"].head(10))
plt.gca().invert_yaxis()
plt.title("Top 10 Positive Lasso Coefficients")
plt.xlabel("Coefficient Value")
plt.show()
