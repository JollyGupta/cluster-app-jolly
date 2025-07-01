import streamlit as st
import pandas as pd
import joblib

# Load models
clf = joblib.load('cluster_predictor.pkl')
cluster_regressors = joblib.load('per_cluster_RF_regressors.pkl')

# Cluster-to-group mapping
cluster_to_group = {
    0: 'Average Customers',
    1: 'Premium Customers',
    2: 'Young High Spenders',
    3: 'Cautious Wealthy',
    4: 'Loyal Customers'
}

# Smaller styled title
st.markdown("<h4 style='text-align:center; color:#333;'>Clustering Project By Jolly_1_July_2025,<br>Base File → My ML Mentor From Learnbay</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#4B8BBE;'>Cluster + Random Forest Regressor Prediction</h4>", unsafe_allow_html=True)

# Mode selector
mode = st.radio("Choose Mode", ["Single Prediction", "Batch (CSV Upload)"])

# -------------------- SINGLE PREDICTION --------------------
if mode == "Single Prediction":
    st.subheader("Step 1: Cluster Prediction Inputs")
    income = st.number_input("Annual Income (k$)", 0.0)
    score = st.number_input("Spending Score (1-100)", 0.0, 100.0)

    st.subheader("Step 2: Regressor Prediction Inputs")
    age = st.number_input("Age", 0, 100)
    genre = st.selectbox("Genre", ["Male", "Female"])

    if st.button("Predict"):
        cluster = clf.predict([[income, score]])[0]
        group = cluster_to_group.get(cluster, "Unknown")
        model = cluster_regressors.get(cluster)

        if model:
            input_df = pd.DataFrame([[age, income, genre]], columns=["Age", "Annual Income (k$)", "Genre"])
            prediction = model.predict(input_df)[0]

            st.success(f"Cluster: {cluster} ({group})")
            st.success(f"Random Forest Prediction: {prediction:.2f}")
        else:
            st.error("Regressor not found for this cluster.")

# -------------------- BATCH PREDICTION --------------------
else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("CSV must contain: Age, Annual Income (k$), Spending Score (1-100), Genre", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Genre"]

        if all(col in df.columns for col in required_cols):
            df["Predicted Cluster"] = clf.predict(df[["Annual Income (k$)", "Spending Score (1-100)"]])
            df["Group"] = df["Predicted Cluster"].map(cluster_to_group)

            predictions = []
            for _, row in df.iterrows():
                model = cluster_regressors.get(row["Predicted Cluster"])
                if model:
                    row_input = pd.DataFrame([row[["Age", "Annual Income (k$)", "Genre"]]])
                    pred = model.predict(row_input)[0]
                else:
                    pred = None
                predictions.append(pred)

            df["Random Forest Prediction"] = predictions
            st.success("Batch prediction completed!")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV with Predictions", csv, "jolly_rf_predictions.csv", "text/csv")
        else:
            st.error(f"Missing columns! Required: {required_cols}")
