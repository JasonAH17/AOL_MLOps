import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.set_page_config(
    page_title="Credit Default Prediction",
    layout="centered"
)

st.title("💳 Credit Card Default Prediction")
st.write(
    "Aplikasi ini memprediksi risiko **gagal bayar kartu kredit** "
    "menggunakan **Best Random Forest Model (Inference Pipeline)**."
)

st.divider()

st.subheader("📥 Masukkan Data Nasabah")

LIMIT_BAL = st.number_input("LIMIT_BAL (Credit Limit)", min_value=0.0)

SEX = st.selectbox("SEX (1 = Male, 2 = Female)", [1, 2])
EDUCATION = st.selectbox("EDUCATION", [1, 2, 3, 4])
MARRIAGE = st.selectbox("MARRIAGE", [1, 2, 3])

AGE = st.number_input("AGE", min_value=18, max_value=100)

st.markdown("### Status Pembayaran (PAY_0 s/d PAY_6)")
PAY_0 = st.number_input("PAY_0", min_value=-2, max_value=8)
PAY_2 = st.number_input("PAY_2", min_value=-2, max_value=8)
PAY_3 = st.number_input("PAY_3", min_value=-2, max_value=8)
PAY_4 = st.number_input("PAY_4", min_value=-2, max_value=8)
PAY_5 = st.number_input("PAY_5", min_value=-2, max_value=8)
PAY_6 = st.number_input("PAY_6", min_value=-2, max_value=8)

st.markdown("### Tagihan Bulanan (BILL_AMT)")
BILL_AMT1 = st.number_input("BILL_AMT1")
BILL_AMT2 = st.number_input("BILL_AMT2")
BILL_AMT3 = st.number_input("BILL_AMT3")
BILL_AMT4 = st.number_input("BILL_AMT4")
BILL_AMT5 = st.number_input("BILL_AMT5")
BILL_AMT6 = st.number_input("BILL_AMT6")

st.markdown("### Pembayaran Bulanan (PAY_AMT)")
PAY_AMT1 = st.number_input("PAY_AMT1")
PAY_AMT2 = st.number_input("PAY_AMT2")
PAY_AMT3 = st.number_input("PAY_AMT3")
PAY_AMT4 = st.number_input("PAY_AMT4")
PAY_AMT5 = st.number_input("PAY_AMT5")
PAY_AMT6 = st.number_input("PAY_AMT6")

if st.button("🔮 Predict Default Risk"):

    # Buat DataFrame sesuai dengan training data
    input_df = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": PAY_3,
        "PAY_4": PAY_4,
        "PAY_5": PAY_5,
        "PAY_6": PAY_6,
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
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ **Berpotensi DEFAULT**\n\n"
            f"Probabilitas Default: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ **TIDAK DEFAULT**\n\n"
            f"Probabilitas Non-Default: **{1 - probability:.2f}**"
        )

st.divider()
st.caption(
    "Model: Random Forest (Inference Pipeline without SMOTE) | "
    "Project: MLOps Credit Default Prediction"
)
