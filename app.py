
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Electric Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

# Load the saved model
model = joblib.load("model.pkl")

# Application title
st.title("⚡ AC Electric Bill Prediction")

st.write(
    "Enter the AC Units to predict your electricity bill."
)

st.divider()

# Input field
ac_units = st.number_input(
    "Enter AC Units",
    min_value=0.0,
    value=10.0,
    step=1.0
)

# Prediction button
if st.button("Predict Electric Bill"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    electric_bill = prediction[0]

    st.success(
        f"Predicted Electric Bill: ₹{electric_bill:.2f}"
    )

    st.info(
        f"AC Units: {ac_units}"
    )
