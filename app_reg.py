import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Regression Models", layout="centered")

# ---------------- Load CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- Title ----------------
st.markdown("""
<div class="card">
    <h1>Regression Models on Tips Dataset</h1>
    <p>Predict <b>Tip Amount</b> using different Regression Techniques</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- Model Selection ----------------
st.subheader("Select Regression Model")

model_choice = st.selectbox(
    "Choose Regression Technique",
    [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ]
)

# ---------------- Feature Selection ----------------
if model_choice == "Simple Linear Regression":
    X = df[["total_bill"]]
else:
    X = df[["total_bill", "size"]]

y = df["tip"]

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Model Initialization ----------------
alpha = None

if model_choice in ["Simple Linear Regression", "Multiple Linear Regression"]:
    model = LinearRegression()

elif model_choice == "Ridge Regression":
    alpha = st.slider("Select Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
    model = Ridge(alpha=alpha)

elif model_choice == "Lasso Regression":
    alpha = st.slider("Select Alpha (Regularization Strength)", 0.01, 10.0, 0.5)
    model = Lasso(alpha=alpha)

# ---------------- Train Model ----------------
model.fit(X_train_scaled, y_train)

# ---------------- Predictions ----------------
y_pred = model.predict(X_test_scaled)

# ---------------- Metrics ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

# ---------------- Visualization ----------------
st.subheader("Total Bill vs Tip Amount")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6, label="Actual")

bill_range = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100)

if model_choice == "Simple Linear Regression":
    X_vis = pd.DataFrame({"total_bill": bill_range})
else:
    X_vis = pd.DataFrame({
        "total_bill": bill_range,
        "size": [df["size"].mean()] * 100
    })

X_vis_scaled = scaler.transform(X_vis)
y_line = model.predict(X_vis_scaled)

ax.plot(bill_range, y_line, color="red", linewidth=2, label="Regression Line")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.legend()

st.pyplot(fig)

# ---------------- Metrics Display ----------------
st.subheader("Model Performance Metrics")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

# ---------------- Coefficients ----------------
st.markdown("""
<div class="card">
    <h3>Model Parameters</h3>
""", unsafe_allow_html=True)

st.write("**Intercept:**", round(model.intercept_, 3))

for i, col in enumerate(X.columns):
    st.write(f"**Coefficient ({col}):**", round(model.coef_[i], 3))

if alpha:
    st.write("**Alpha:**", alpha)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction Section ----------------
st.subheader("Predict Tip Amount")

bill_input = st.slider(
    "Total Bill ($)",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    25.0
)

if model_choice == "Simple Linear Regression":
    input_data = scaler.transform([[bill_input]])
else:
    size_input = st.slider(
        "Table Size",
        int(df["size"].min()),
        int(df["size"].max()),
        2
    )
    input_data = scaler.transform([[bill_input, size_input]])

predicted_tip = model.predict(input_data)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip Amount: ${predicted_tip:.2f}</div>',
    unsafe_allow_html=True
)