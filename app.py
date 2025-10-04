import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.title("📊 پلتفرم بهینه‌سازی پرتفوی (Markowitz)")

st.markdown("سناریوها، احتمال‌ها و بازدهی دارایی‌ها را وارد کنید:")

# -------------------------------
# ورودی: سناریوها
# -------------------------------
num_scenarios = st.number_input("تعداد سناریوها", min_value=1, max_value=10, value=2)

scenarios = []
for i in range(num_scenarios):
    st.subheader(f"سناریو {i+1}")
    prob = st.number_input(f"🔹 احتمال وقوع سناریو {i+1}", min_value=0.0, max_value=1.0, value=0.5, key=f"p{i}")
    stock = st.number_input(f"بازده سهام در سناریو {i+1}", value=0.0, format="%.3f", key=f"s{i}")
    bond = st.number_input(f"بازده درآمد ثابت در سناریو {i+1}", value=0.0, format="%.3f", key=f"b{i}")
    gold = st.number_input(f"بازده طلا در سناریو {i+1}", value=0.0, format="%.3f", key=f"g{i}")
    scenarios.append([prob, stock, bond, gold])

scenarios = np.array(scenarios)
probs = scenarios[:, 0]
returns = scenarios[:, 1:]

# -------------------------------
# ورودی: محدودیت‌ها
# -------------------------------
st.header("🔹 محدودیت‌های وزن دارایی‌ها")

min_stock = st.number_input("حداقل وزن سهام", 0.0, 1.0, 0.25)
max_stock = st.number_input("حداکثر وزن سهام", 0.0, 1.0, 0.5)

min_bond = st.number_input("حداقل وزن درآمد ثابت", 0.0, 1.0, 0.3)
max_bond = st.number_input("حداکثر وزن درآمد ثابت", 0.0, 1.0, 0.4)

min_gold = st.number_input("حداقل وزن طلا", 0.0, 1.0, 0.2)
max_gold = st.number_input("حداکثر وزن طلا", 0.0, 1.0, 0.4)

bounds = [(min_stock, max_stock), (min_bond, max_bond), (min_gold, max_gold)]

# -------------------------------
# انتخاب نوع مدل
# -------------------------------
model_choice = st.selectbox(
    "مدل بهینه‌سازی را انتخاب کنید:",
    ["کمینه ریسک (Min Variance)", "بیشینه بازده (Max Return)", "بیشینه شارپ (Max Sharpe)"]
)

# -------------------------------
# محاسبات
# -------------------------------
expected_returns = returns.T @ probs

def scenario_cov(returns, probs):
    mean = returns.T @ probs
    diff = returns - mean
    weighted_diff = diff.T * probs
    return weighted_diff @ diff

cov_matrix = scenario_cov(returns, probs)

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
x0 = np.array([1/3, 1/3, 1/3])

def objective_min_variance(weights):
    return weights.T @ cov_matrix @ weights

def negative_expected_return(weights):
    return -weights @ expected_returns

def negative_sharpe(weights, risk_free=0.0):
    port_return = expected_returns @ weights
    port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
    return -(port_return - risk_free) / port_risk if port_risk > 0 else 1e6

# -------------------------------
# اجرای انتخاب شده توسط کاربر
# -------------------------------
if st.button("📈 محاسبه پرتفوی"):
    if model_choice == "کمینه ریسک (Min Variance)":
        result = minimize(objective_min_variance, x0, bounds=bounds, constraints=constraints)
    elif model_choice == "بیشینه بازده (Max Return)":
        result = minimize(negative_expected_return, x0, bounds=bounds, constraints=constraints)
    elif model_choice == "بیشینه شارپ (Max Sharpe)":
        result = minimize(negative_sharpe, x0, bounds=bounds, constraints=constraints)

    if result.success:
        w = result.x
        port_return = expected_returns @ w
        port_risk = np.sqrt(w.T @ cov_matrix @ w)
        sharpe = (port_return) / port_risk if port_risk > 0 else 0

        st.subheader("📊 نتایج پرتفوی بهینه")
        df = pd.DataFrame({
            "دارایی": ["سهام", "درآمد ثابت", "طلا"],
            "وزن": w.round(3)
        })
        st.table(df)

        st.write(f"🔹 بازده مورد انتظار پرتفوی: {port_return:.3f}")
        st.write(f"🔹 ریسک پرتفوی: {port_risk:.3f}")
        st.write(f"🔹 نسبت شارپ: {sharpe:.3f}")

        # -------------------------------
        # 📊 نمودارها
        # -------------------------------
        st.subheader("📊 نمودار سهم دارایی‌ها (Pie Chart)")
        fig1, ax1 = plt.subplots()
        ax1.pie(w, labels=["سهام", "درآمد ثابت", "طلا"], autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        st.subheader("📊 نمودار ریسک و بازده پرتفوی")
        fig2, ax2 = plt.subplots()
        ax2.bar(["بازده", "ریسک"], [port_return, port_risk], color=["green", "red"])
        ax2.set_ylabel("مقدار")
        st.pyplot(fig2)

    else:
        st.error("❌ بهینه‌سازی موفق نبود.")
