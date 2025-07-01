import streamlit as st
import pandas as pd
import joblib

# Load models
clf = joblib.load('cluster_predictor.pkl')
cluster_regressors = joblib.load('per_cluster_regressors.pkl')

cluster_to_group = {
    0: 'Average Customers',
    1: 'Premium Customers',
    2: 'Young High Spenders',
    3: 'Cautious Wealthy',
    4: 'Loyal Customers'
}

# Styled titles
st.markdown("<h4 style='text-align:center; color:#333;'>Clustering Project By Jolly_1_July_2025,<br>Base File → My ML Mentor From Learnbay</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#4B8BBE;'>Cluster + Regression Prediction</h4>", unsafe_allow_html=True)

# Mode switch
mode = st.radio("Choose Mode", ["Single Prediction", "Upload CSV for Multiple Predictions"])

# ---------------------- SINGLE PREDICTION ----------------------
if mode == "Single Prediction":
    st.header("Step 1: Predict Cluster")
    income = st.number_input("Annual Income (k$)", 0.0)
    score = st.number_input("Spending Score (1-100)", 0.0, 100.0)

    st.header("Step 2: Predict from Regressor")
    age = st.number_input("Age", 0, 100)
    genre = st.selectbox("Genre", ["Male", "Female"])

    if st.button("Predict_By_Jolly"):
        cluster = clf.predict([[income, score]])[0]
        group = cluster_to_group.get(cluster, "Unknown")
        model = cluster_regressors.get(cluster)

        if model:
            df_input = pd.DataFrame([[age, income, genre]], columns=["Age", "Annual Income (k$)", "Genre"])
            result = model.predict(df_input)[0]
            st.success(f"Cluster: {cluster} ({group})")
            st.success(f"Prediction from Regressor: {result:.2f}")
        else:
            st.error("Regressor not found for this cluster.")

# ---------------------- MULTIPLE PREDICTIONS ----------------------
else:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file with columns: Age, Annual Income (k$), Spending Score (1-100), Genre", type=["csv"])

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

            df["Regressor Prediction"] = predictions
            st.success("✅ Predictions complete!")
            st.dataframe(df.head())

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "jolly_predictions.csv", "text/csv")
        else:
            st.error(f"CSV must contain: {required_cols}")
