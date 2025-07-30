
import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="Aircrash Severity Predictor", layout="centered")
st.title("✈️ Aircrash Severity Predictor")
st.markdown("""
This web app uses a trained machine learning model to predict whether an airplane crash is likely to be **severe** or **not severe**, based on key information such as the number of people aboard and crash location details.
""")

# Load trained model
try:
    with open("aircrash_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'aircrash_model.pkl' is in the same directory.")
    st.stop()

# --------------------------------------
# 🔹 Single Prediction Section
# --------------------------------------
st.header("📥 Enter Crash Details")

with st.form("prediction_form"):
    operator = st.text_input("Operator Name (e.g., Delta, Military - US Navy)")
    aircraft_type = st.text_input("Aircraft Type (e.g., Boeing 737, Zeppelin)")
    location = st.text_input("Crash Location (City, Region, etc.)")
    route = st.text_input("Flight Route (e.g., NYC to LA)")
    aboard = st.number_input("Total People Aboard", min_value=1, value=50)
    ground = st.number_input("People Affected on Ground", min_value=0, value=0)
    submitted = st.form_submit_button("Predict Severity")

if submitted:
    input_df = pd.DataFrame({
        "Operator": [hash(operator) % 1000],
        "AC Type": [hash(aircraft_type) % 1000],
        "Location": [hash(location) % 1000],
        "Route": [hash(route) % 1000],
        "Aboard": [aboard],
        "Ground": [ground]
    })

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"🟥 Prediction: **Severe Crash** (Confidence: {prob:.2f})")
    else:
        st.success(f"🟩 Prediction: **Not Severe** (Confidence: {prob:.2f})")

# --------------------------------------
# 📤 Bulk Prediction Section
# --------------------------------------
st.header("📁 Bulk Prediction from CSV")

st.markdown("""
Upload a CSV file with the following columns:
- Operator
- AC Type
- Location
- Route
- Aboard
- Ground
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        required_cols = ['Operator', 'AC Type', 'Location', 'Route', 'Aboard', 'Ground']
        if not all(col in input_df.columns for col in required_cols):
            st.error(f"❌ CSV must contain the following columns: {', '.join(required_cols)}")
        else:
            for col in ['Operator', 'AC Type', 'Location', 'Route']:
                input_df[col] = input_df[col].astype(str).apply(lambda x: hash(x) % 1000)

            preds = model.predict(input_df)
            probs = model.predict_proba(input_df).max(axis=1)

            input_df["Severity_Prediction"] = ["Severe" if p == 1 else "Not Severe" for p in preds]
            input_df["Confidence"] = probs.round(2)

            st.success("✅ Bulk predictions completed.")
            st.dataframe(input_df)

            csv_download = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv_download, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# Footer
st.markdown("---")
st.caption("Developed as part of A8 Assignment | Streamlit | Machine Learning | 2025")
