import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("cluster_predictor.pkl")
features = ["Annual Income (k$)", "Spending Score (1-100)"]
cluster_to_group = {
    0: 'Average Customers',
    1: 'Premium Customers',
    2: 'Young High Spenders',
    3: 'Cautious Wealthy',
    4: 'Loyal Customers'
}

st.title("Customer Cluster Predictor")
mode = st.radio("Choose Mode", ["Single", "CSV Upload"])

if mode == "Single":
    income = st.number_input("Annual Income (k$)", 0.0)
    score = st.number_input("Spending Score (1-100)", 0.0, 100.0)
    if st.button("Predict"):
        cluster = model.predict([[income, score]])[0]
        st.success(f"Cluster: {cluster} ({cluster_to_group.get(cluster)})")

else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if all(col in df.columns for col in features):
            df["Predicted Cluster"] = model.predict(df[features])
            df["Group"] = df["Predicted Cluster"].map(cluster_to_group)
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "predicted_clusters.csv")
        else:
            st.error(f"CSV must contain: {features}")

