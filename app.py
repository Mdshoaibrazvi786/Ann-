import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Smart Campus Density Predictor", layout="wide")
st.title("Smart Campus Density Index Prediction")

# Model load handler
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('smart_campus_model.keras')

try:
    model = load_keras_model()
    st.success("Model Successfully Loaded!")
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Dataset ke 25 Features
feature_cols = [
    'hour_index', 'day_index', 'week_index', 'space_id', 'capacity',
    'occupancy_count', 'utilization_ratio', 'idle_minutes', 'power_kwh',
    'hvac_state', 'voltage_variation', 'device_temperature', 'zone_id',
    'connected_devices', 'bandwidth_usage', 'packet_loss_rate',
    'corridor_id', 'entry_count', 'exit_count', 'feature_20',
    'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25'
]

st.subheader("Enter Values for Prediction")
input_data = {}
cols = st.columns(3)

for i, col_name in enumerate(feature_cols):
    with cols[i % 3]:
        input_data[col_name] = st.number_input(f"{col_name}", value=0.0)

if st.button("Predict Density Index"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0][0]
    st.success(f"Predicted Density Index: **{prediction:.2f}**")
