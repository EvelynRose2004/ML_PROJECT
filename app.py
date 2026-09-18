
import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures

# Page configuration
st.set_page_config(
    page_title="AC Electric Bill Prediction",
    page_icon="⚡",
    layout="centered"
)

# Load the saved model
model = joblib.load("model.pkl")

# Create PolynomialFeatures
poly = PolynomialFeatures(degree=2)

# Application title
st.title("⚡ AC Electric Bill Prediction")

st.write("Enter the AC Units to predict your electricity bill.")

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

    # Transform input into polynomial features
    input_poly = poly.fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_poly)

    # Get predicted electric bill
    electric_bill = prediction[0]

    # Display result
    st.success(
        f"Predicted Electric Bill: ₹{electric_bill:.2f}"
    )

    st.info(
        f"AC Units Entered: {ac_units}"
    )
