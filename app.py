import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")


@st.cache_resource
def load_bundle():
    return joblib.load("churn_model.joblib")


bundle = load_bundle()
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]

st.title("📉 Telco Customer Churn Predictor")
st.write("Customer ki details bharo, model batayega ke wo company chhorne ka risk rakhta hai ya nahi.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0, step=0.5)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    phone = st.selectbox("Phone service", ["Yes", "No"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

if phone == "No":
    multiple_lines = "No phone service"
else:
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

if internet == "No":
    addon = "No internet service"
    online_security = online_backup = device_protection = addon
    tech_support = streaming_tv = streaming_movies = addon
    st.caption("Internet nahi hai, isliye add-on services 'No internet service' hain.")
else:
    st.subheader("Add-on services")
    c1, c2, c3 = st.columns(3)
    online_security = c1.selectbox("Online security", ["No", "Yes"])
    online_backup = c2.selectbox("Online backup", ["No", "Yes"])
    device_protection = c3.selectbox("Device protection", ["No", "Yes"])
    tech_support = c1.selectbox("Tech support", ["No", "Yes"])
    streaming_tv = c2.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = c3.selectbox("Streaming movies", ["No", "Yes"])

if st.button("Predict", type="primary"):
    raw = pd.DataFrame([{
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }])

    encoded = pd.get_dummies(raw).reindex(columns=columns, fill_value=0).astype(float)
    scaled = scaler.transform(encoded)

    prob = model.predict_proba(scaled)[0][1]

    st.subheader("Result")
    st.metric("Churn probability", f"{prob:.0%}")
    st.progress(float(prob))
    if prob >= 0.5:
        st.error("⚠️ High churn risk — is customer ko retain karne ki koshish karo.")
    else:
        st.success("✅ Low churn risk.")
    st.caption(
        "Model ko class_weight='balanced' pe train kiya gaya hai, isliye probabilities "
        "churn ki taraf thori zyada jhuki hoti hain. Isko risk score samjho, exact chance nahi."
    )
