# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_data, prepare_features, haversine_distance
from model import train_and_evaluate, save_model, load_model

st.set_page_config(page_title="Uber Fare Prediction", layout="wide")
st.title("Uber Fare Prediction — Streamlit Web App")

# Sidebar controls
st.sidebar.header("Project Controls")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type=['csv'])
use_sample = st.sidebar.checkbox("Load sample uber.csv")

model_dir = "models"
model_path = os.path.join(model_dir, "uber_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Load data
df = None
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded dataset loaded.")
elif use_sample:
    df = load_data("uber.csv")
    if df is None:
        st.sidebar.error("uber.csv not found.")

# Tabs
tabs = st.tabs(["Overview & EDA", "Train Model", "Predict", "Project Notes"])

# ----------------------- TAB 1: EDA -----------------------
with tabs[0]:
    st.header("Dataset Overview")
    if df is None:
        st.info("Upload or load a dataset to begin.")
    else:
        st.subheader("Sample rows")
        st.dataframe(df.head())

        st.subheader("Statistics")
        st.write(df.describe(include="all"))

        st.subheader("Column Types")
        st.write(df.dtypes)

        # Plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col = st.selectbox("Select numeric column", numeric_cols)

        if col:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df[col], bins=40, edgecolor="black")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

# ----------------------- TAB 2: Train Model -----------------------
with tabs[1]:
    st.header("Train a Model")
    if df is None:
        st.info("Load data first.")
    else:
        target = st.selectbox("Target Column", df.columns)

        X, y, features = prepare_features(df, target_col=target)
        st.write("Features detected:", features)

        model_choice = st.selectbox("Model Type", ["RandomForest", "GradientBoosting", "LinearRegression"])

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                result = train_and_evaluate(X, y, model_choice)
                save_model(result["model"], result["scaler"], model_path, scaler_path)

                st.success("Model trained and saved!")
                st.metric("MAE", f"{result['mae']:.3f}")
                st.metric("RMSE", f"{result['rmse']:.3f}")
                st.metric("R2 Score", f"{result['r2']:.3f}")

                if hasattr(result["model"], "feature_importances_"):
                    st.subheader("Feature Importances")
                    st.write(
                        pd.DataFrame({
                            "feature": features,
                            "importance": result["model"].feature_importances_
                        }).sort_values("importance", ascending=False)
                    )

# ----------------------- TAB 3: Predict -----------------------
with tabs[2]:
    st.header("Make Predictions")

    model_loaded, scaler_loaded = load_model(model_path, scaler_path)

    if model_loaded is None:
        st.error("Train a model first.")
    else:
        st.subheader("Single Prediction")

        col1, col2 = st.columns(2)
        with col1:
            pickup_lat = st.number_input("Pickup Latitude", value=40.7614)
            pickup_lon = st.number_input("Pickup Longitude", value=-73.9798)
            drop_lat = st.number_input("Dropoff Latitude", value=40.6413)
            drop_lon = st.number_input("Dropoff Longitude", value=-73.7781)

        with col2:
            passenger_count = st.number_input("Passenger Count", 1, 8, 1)
            dt = st.datetime_input("Pickup Datetime")

        if st.button("Predict Fare"):
            dist = haversine_distance(pickup_lat, pickup_lon, drop_lat, drop_lon)
            hour = dt.hour
            dow = dt.weekday()

            X_pred = pd.DataFrame([[dist, hour, dow, passenger_count]],
                                  columns=["distance_km", "hour", "dayofweek", "passenger_count"])

            Xs = scaler_loaded.transform(X_pred)
            pred = model_loaded.predict(Xs)[0]

            st.success(f"Predicted Fare: ${pred:.2f}")

        st.subheader("Batch Prediction (CSV)")
        batch = st.file_uploader("Upload CSV", type=["csv"], key="batch")

        if batch:
            dfb = pd.read_csv(batch)
            Xb, _, feats = prepare_features(dfb, target_col=None)

            if len(feats) == 0:
                st.error("Features not detected in CSV.")
            else:
                Xb_s = scaler_loaded.transform(Xb)
                dfb["predicted_fare"] = model_loaded.predict(Xb_s)
                st.dataframe(dfb.head())

                st.download_button("Download predictions", dfb.to_csv(index=False), "predictions.csv")


# ----------------------- TAB 4: Notes -----------------------
with tabs[3]:
    st.header("Project Notes")
    st.write("""
    - End-to-end ML pipeline  
    - Streamlit frontend  
    - Feature engineering (distance, datetime)  
    - Multiple ML models supported  
    - Saved model inference  
    """)
