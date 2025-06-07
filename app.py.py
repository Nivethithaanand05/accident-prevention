import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Define required input features
required_features = ['speed', 'jerk_score', 'overspeed_flag', 'control_loss_flag']

# Streamlit UI
st.title("🚦 AI-Driven Road Safety Prediction")
st.markdown("Upload vehicle behavior data to predict accident risk levels.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV file", type="csv")

if uploaded_file is not None:
    # Load and preview data
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(data.head())

    # Check if required features are present
    if all(feature in data.columns for feature in required_features):
        # Filter to required features
        input_data = data[required_features]

        # Make predictions
        predictions = model.predict(input_data)

        # Add predictions to original data
        data['Accident Risk'] = ['⚠️ High Risk' if p == 1 else '✅ Low Risk' for p in predictions]

        # Display final result
        st.subheader("🧠 Prediction Results")
        st.dataframe(data[['speed', 'jerk_score', 'overspeed_flag', 'control_loss_flag', 'Accident Risk']])
    else:
        missing_cols = [f for f in required_features if f not in data.columns]
        st.error(f"❌ Missing required columns: {missing_cols}")
