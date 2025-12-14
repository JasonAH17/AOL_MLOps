import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Credit Default Prediction",
    layout="centered"
)

st.title("💳 Credit Card Default Prediction")
st.write(
    "This application predicts the **risk of credit card default** "
    "based on customer demographic and payment behavior."
)

st.divider()

# =========================
# MAPPINGS
# =========================

gender_map = {"Male": 1, "Female": 2}

education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Other": 4
}

marriage_map = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}

payment_status_map = {
    "Paid on time": 0,
    "Paid in full": -1,
    "No consumption": -2,
    "1 month late": 1,
    "2 months late": 2,
    "3 months late": 3,
    "4+ months late": 4
}

# =========================
# CUSTOMER INFORMATION
# =========================

st.subheader("👤 Customer Information")
col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input(
        "Credit Limit",
        min_value=0.0,
        help="Total credit limit granted to the customer."
    )

    SEX = st.selectbox(
        "Gender",
        options=list(gender_map.keys()),
        help="Customer gender."
    )

    EDUCATION = st.selectbox(
        "Education Level",
        options=list(education_map.keys()),
        help="Highest education level attained."
    )

with col2:
    AGE = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        help="Customer age in years."
    )

    MARRIAGE = st.selectbox(
        "Marital Status",
        options=list(marriage_map.keys()),
        help="Customer marital status."
    )

st.divider()

# =========================
# PAYMENT STATUS
# =========================

st.subheader("📅 Payment Status History")
col1, col2 = st.columns(2)

with col1:
    PAY_0 = st.selectbox(
        "Payment Status (September)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in September."
    )
    PAY_2 = st.selectbox(
        "Payment Status (August)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in August."
    )
    PAY_3 = st.selectbox(
        "Payment Status (July)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in July."
    )

with col2:
    PAY_4 = st.selectbox(
        "Payment Status (June)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in June."
    )
    PAY_5 = st.selectbox(
        "Payment Status (May)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in May."
    )
    PAY_6 = st.selectbox(
        "Payment Status (April)",
        options=list(payment_status_map.keys()),
        help="Payment behavior in April."
    )

st.divider()

# =========================
# BILL & PAYMENT AMOUNTS
# =========================

st.subheader("💳 Billing & Payment Amounts")
col1, col2 = st.columns(2)

with col1:
    BILL_AMT1 = st.number_input("Bill Amount (September)", help="Bill in September.")
    BILL_AMT2 = st.number_input("Bill Amount (August)")
    BILL_AMT3 = st.number_input("Bill Amount (July)")

    PAY_AMT1 = st.number_input("Payment Amount (September)", help="Payment in September.")
    PAY_AMT2 = st.number_input("Payment Amount (August)")
    PAY_AMT3 = st.number_input("Payment Amount (July)")

with col2:
    BILL_AMT4 = st.number_input("Bill Amount (June)")
    BILL_AMT5 = st.number_input("Bill Amount (May)")
    BILL_AMT6 = st.number_input("Bill Amount (April)")

    PAY_AMT4 = st.number_input("Payment Amount (June)")
    PAY_AMT5 = st.number_input("Payment Amount (May)")
    PAY_AMT6 = st.number_input("Payment Amount (April)")

st.divider()

# =========================
# PREDICTION
# =========================

if st.button("🔮 Predict Default Risk"):

    # ===== FEATURE ENGINEERING =====
    TOTAL_BILL_AMT = (
        BILL_AMT1 + BILL_AMT2 + BILL_AMT3 +
        BILL_AMT4 + BILL_AMT5 + BILL_AMT6
    )

    TOTAL_PAY_AMT = (
        PAY_AMT1 + PAY_AMT2 + PAY_AMT3 +
        PAY_AMT4 + PAY_AMT5 + PAY_AMT6
    )

    PAYMENT_RATIO = TOTAL_PAY_AMT / (TOTAL_BILL_AMT + 1)

    HAS_MISSED_PAYMENT = int(
        any(status > 0 for status in [
            payment_status_map[PAY_0],
            payment_status_map[PAY_2],
            payment_status_map[PAY_3],
            payment_status_map[PAY_4],
            payment_status_map[PAY_5],
            payment_status_map[PAY_6],
        ])
    )

    # ===== INPUT DATAFRAME =====
    input_df = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": gender_map[SEX],
        "EDUCATION": education_map[EDUCATION],
        "MARRIAGE": marriage_map[MARRIAGE],
        "AGE": AGE,

        "PAY_0": payment_status_map[PAY_0],
        "PAY_2": payment_status_map[PAY_2],
        "PAY_3": payment_status_map[PAY_3],
        "PAY_4": payment_status_map[PAY_4],
        "PAY_5": payment_status_map[PAY_5],
        "PAY_6": payment_status_map[PAY_6],

        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": BILL_AMT2,
        "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4,
        "BILL_AMT5": BILL_AMT5,
        "BILL_AMT6": BILL_AMT6,

        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": PAY_AMT2,
        "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4,
        "PAY_AMT5": PAY_AMT5,
        "PAY_AMT6": PAY_AMT6,

        # ENGINEERED FEATURES
        "TOTAL_BILL_AMT": TOTAL_BILL_AMT,
        "TOTAL_PAY_AMT": TOTAL_PAY_AMT,
        "PAYMENT_RATIO": PAYMENT_RATIO,
        "HAS_MISSED_PAYMENT": HAS_MISSED_PAYMENT,
    }])

    # ===== PREDICT =====
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ **High Risk of Default**\n\n"
            f"Probability of Default: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ **Low Risk of Default**\n\n"
            f"Probability of Non-Default: **{1 - probability:.2f}**"
        )


st.divider()
st.caption(
    "Dataset: UCI Default of Credit Card Clients | "
    "Model: Random Forest (Inference Pipeline) | "
    "Project: MLOps Credit Risk Prediction"
)
