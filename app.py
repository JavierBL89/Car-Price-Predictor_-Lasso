import streamlit as st
import pandas as pd
import statsmodels.api as sm
import pickle
import joblib
import sklearn


# Load model and scaler
lasso = joblib.load("models/lasso_car_price_model.pkl")
scaler = joblib.load("models/scaler_car_price.pkl")
feature_cols = joblib.load("models/lasso_feature_columns.pkl")

st.title("Car Price Predictor")
st.markdown("""
This app predicts the **car price** based on its technical and brand features.  
It was trained on a dataset of 205 cars, using **Multiple Linear Regression**.  
All redundant and correlated features were removed after VIF and OLS analysis.
The final training data set went down to 11 features from the original 67.
""")


st.divider()

# -----------------------------
# 🎛️ Two-column layout
# -----------------------------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    # --- User inputs ---
    st.write("Enter car details below to predict the estimated price.")
    carwidth = st.slider("Car Width (inches)", 60.0, 75.0, 68.0)
    citympg = st.slider("City MPG", 10, 50, 25)
    horsepower = st.slider("Horsepower", 50, 300, 150, step=10)

    # Categorical inputs (match dataset values!)
    carbody = st.selectbox("Car Body Type", ["hardtop", "hatchback", "sedan", "wagon"])
    enginetype = st.selectbox("Engine Type", ["ohc", "ohcv"])  # lowercase to match model
    brand = st.selectbox(
        "Brand",
        ["audi", "bmw", "buick", "dodge", "honda", "isuzu", "jaguar", "mazda",
         "mitsubishi", "nissan", "plymouth", "renault", "saab", "toyota",
         "volkswagen", "volvo"]
    )
    drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd"])

     # --- Convert user input to model input ---
    input_data = pd.DataFrame({
        "carwidth": [carwidth],
        "citympg": [citympg],
        "horsepower": [horsepower],
        "carbody_hardtop": [1 if carbody == "hardtop" else 0],
        "carbody_hatchback": [1 if carbody == "hatchback" else 0],
        "carbody_sedan": [1 if carbody == "sedan" else 0],
        "carbody_wagon": [1 if carbody == "wagon" else 0],
        "enginetype_ohc": [1 if enginetype == "ohc" else 0],
        "enginetype_ohcv": [1 if enginetype == "ohcv" else 0],
        "drivewheel_rwd": [1 if drivewheel == "rwd" else 0],

        # --- Brand dummies (one-hot) ---
        "maker_audi": [1 if brand == "audi" else 0],
        "maker_bmw": [1 if brand == "bmw" else 0],
        "maker_buick": [1 if brand == "buick" else 0],
        "maker_dodge": [1 if brand == "dodge" else 0],
        "maker_honda": [1 if brand == "honda" else 0],
        "maker_isuzu": [1 if brand == "isuzu" else 0],
        "maker_jaguar": [1 if brand == "jaguar" else 0],
        "maker_mazda": [1 if brand == "mazda" else 0],
        "maker_mitsubishi": [1 if brand == "mitsubishi" else 0],
        "maker_nissan": [1 if brand == "nissan" else 0],
        "maker_plymouth": [1 if brand == "plymouth" else 0],
        "maker_renault": [1 if brand == "renault" else 0],
        "maker_saab": [1 if brand == "saab" else 0],
        "maker_toyota": [1 if brand == "toyota" else 0],
        "maker_volkswagen": [1 if brand == "volkswagen" else 0],
        "maker_volvo": [1 if brand == "volvo" else 0]
    })


    # Ensure all columns align with training features
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[feature_cols]
    # Scale input
    scaled_input = scaler.transform(input_data)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    if st.button("🔮 Predict Price"):
        prediction = lasso.predict(scaled_input)[0]
        st.success(f"💰 Estimated Price: **${prediction:,.2f}**")



# 🧠 Educational Panel (Updated for Lasso Model)
with right_col:
    st.markdown("## 📘 Educational Insight")

    st.markdown("### How this model works")
    st.markdown("""
    - **Type:** Regularized Multiple Linear Regression (Lasso)
    - **Goal:** Predict car price based on performance and brand features  
    - **Final R²:** 0.871 → explains **~87% of price variance**  
    - **RMSE:** ≈ 2760 → average prediction error ≈ **$2,760**
    """)

    st.markdown("### 🔑 Key Factors")
    st.markdown("""
    - **Horsepower:** strongest positive influence — more power, higher price  
    - **Drive Type (RWD):** adds performance value  
    - **Brand:** BMW, Jaguar, Buick, and Volvo models have higher price trends  
    - **Engine Type (OHCV):** performance-oriented engines slightly increase price  
    - **Car Body Type:** sedans and wagons generally priced higher than hatchbacks
    """)

    st.markdown("### ⚠️ Why prices may differ")
    st.markdown("""
    - Model reflects **training dataset** (not current market values).  
    - Some brands (like **Porsche** or **Saab**) could appear cheaper due to having smaller or older models in the dataset.  
    - Regression captures **average relationships**, not brand prestige.  
    """)

    st.info("💡 *Regularization (Lasso)* automatically removes weak predictors, keeping only the most relevant features for stable predictions.")
