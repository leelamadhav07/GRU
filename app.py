import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = tf.keras.models.load_model("models/gru_model.keras")

# -----------------------------------
# LOAD SCALER
# -----------------------------------

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------------
# PAGE
# -----------------------------------

st.title("GRU Passenger Forecasting")

st.write("Predict Next Month Passengers")

values = []

for i in range(12):
    value = st.number_input(f"Month {i + 1}", min_value=0, value=100)

    values.append(value)

# -----------------------------------
# PREDICT
# -----------------------------------

if st.button("Predict"):
    data = np.array(values).reshape(-1, 1)

    data = scaler.transform(data)

    data = data.reshape(1, 12, 1)

    prediction = model.predict(data)

    prediction = scaler.inverse_transform(prediction)

    st.success(f"Predicted Passengers : {prediction[0][0]:.0f}")
