import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config & CSS
st.set_page_config(page_title="Regression Models - Tips Dataset", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
# Title
st.markdown("""
<div class="card">
<h1>Regression Models on Tips Dataset</h1>
<p>Predict <b>Tip Amount</b> using different regression techniques</p>
</div>
""", unsafe_allow_html=True)
# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())
# Sidebar - Model Selection
model_type = st.sidebar.selectbox(
    "Select Regression Model",
    (
        "Multiple Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression"
    )
)
# Data Preprocessing (for prediction models)
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "day", "time"],
    drop_first=True
)

X = df_encoded.drop("tip", axis=1)
y = df_encoded["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model Selection
if model_type == "Multiple Linear Regression":
    model = LinearRegression()

elif model_type == "Polynomial Regression":
    degree = st.sidebar.slider("Polynomial Degree", 2, 4, 2)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)
    model = LinearRegression()

elif model_type == "Ridge Regression":
    alpha = st.sidebar.slider("Alpha (λ)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)

elif model_type == "Lasso Regression":
    alpha = st.sidebar.slider("Alpha (λ)", 0.01, 10.0, 0.1)
    model = Lasso(alpha=alpha)
# Train & Predict
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display Metrics
st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Model", model_type)

# SIMPLE LINEAR REGRESSION (FOR VISUALIZATION ONLY)
st.subheader("Total Bill vs Tip Amount")
x_simple = df[["total_bill"]]
y_simple = df["tip"]
scaler_simple = StandardScaler()
x_simple_scaled = scaler_simple.fit_transform(x_simple)
lin_model = LinearRegression()
lin_model.fit(x_simple_scaled, y_simple)
x_sorted = np.sort(x_simple.values.reshape(-1, 1))
x_sorted_scaled = scaler_simple.transform(x_sorted)
y_line = lin_model.predict(x_sorted_scaled)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(x_sorted, y_line, color="red", linewidth=2)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
st.pyplot(fig)
# Intercept & Coefficient Section (LIKE SCREENSHOT)
st.markdown(
    f"""
    <div class="card">
        <h3>Model intercept and coefficient</h3>
        <p><b>Intercept:</b> {lin_model.intercept_:.2f}</p>
        <p><b>Coefficient for Total Bill:</b> {lin_model.coef_[0]:.2f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Prediction Section (Slider Based)
st.subheader("Prediction on Tip")

bill = st.slider(
    "Total Bill Amount $",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    25.0
)

# Use mean values for other features
input_data = X.mean().to_dict()
input_data["total_bill"] = bill

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if model_type == "Polynomial Regression":
    input_scaled = poly.transform(input_scaled)

predicted_tip = model.predict(input_scaled)[0]

st.markdown(
    f"""
    <div class="prediction-box">
        Predicted Tip Amount: ${predicted_tip:.2f}
    </div>
    """,
    unsafe_allow_html=True
)