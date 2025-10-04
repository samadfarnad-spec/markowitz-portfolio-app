import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.title("📊 پلتفرم بهینه‌سازی پرتفوی (Markowitz)")

st.markdown("سناریوها، احتمال‌ها و بازدهی دارایی‌ها را وارد کنید:")

# -------------------------------
# ورودی: سناریوها و بازدهی
# -------------------------------
num_scenarios = st.number_input("تعداد سناریوها", min_value=1, max_value=10, value=2)

scenarios = []
for i in range(num_scenarios):
    st.subheader(f"سناریو {i+1}")
    prob = st.number_input(f"🔹 احتمال وقوع سناریو {i+1}", min_value=0.0, max_value=1.0, value=0.5)
    stock = st.number_input(f"بازده سهام در سناریو {i+1}", value=0.0, format="%.3f")
    bond = st.number_input(f"بازده درآمد ثابت در سناریو {i+1}", value=0.0, format="%.3f")
    gold = st.number_input(f"بازده طلا در سناریو {i+1}", value=0.0, format="%.3f")
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
# بهینه‌سازی مارکویتز
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

# اجرای مدل‌ها
min_var_res = minimize(objective_min_variance, x0, bounds=bounds, constraints=constraints)
max_ret_res = minimize(negative_expected_return, x0, bounds=bounds, constraints=constraints)

# -------------------------------
# نمایش خروجی
# -------------------------------
if st.button("📈 محاسبه پرتفوی"):
    results = []

    if min_var_res.success:
        w = min_var_res.x
        ret = expected_returns @ w
        risk = np.sqrt(w.T @ cov_matrix @ w)
        results.append(["مارکویتز (کمینه ریسک)", *w, ret, risk])

    if max_ret_res.success:
        w = max_ret_res.x
        ret = expected_returns @ w
        risk = np.sqrt(w.T @ cov_matrix @ w)
        results.append(["بیشینه‌سازی بازده", *w, ret, risk])

    df_results = pd.DataFrame(results, columns=[
        "مدل", "وزن سهام", "وزن درآمد ثابت", "وزن طلا", "بازده مورد انتظار", "ریسک"
    ])
    st.subheader("نتایج بهینه‌سازی")
    st.table(df_results)

    # امکان دانلود خروجی
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 دانلود نتایج (CSV)", data=csv, file_name="results.csv", mime="text/csv")
