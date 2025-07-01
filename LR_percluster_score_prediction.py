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

#st.title("Clustering Project By Jolly_1_July_2025, Base File-> My ML Mentor From Learnbay")
#st.title("Cluster + Regression Prediction")

st.markdown("<h4 style='text-align:center; color:#333;'>Clustering Project By Jolly_1_July_2025,<br>Base File → My ML Mentor From Learnbay</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#4B8BBE;'>Cluster + Regression Prediction</h4>", unsafe_allow_html=True)


# Step 1: Cluster prediction inputs
st.header("Step 1: Predict Cluster")
income = st.number_input("Annual Income (k$)", 0.0)
score = st.number_input("Spending Score (1-100)", 0.0, 100.0)

# Step 2: Regression inputs
st.header("Step 2: Predict from Regressor")
age = st.number_input("Age", 0, 100)
genre = st.selectbox("Genre", ["Male", "Female"])

if st.button("Predict_By_Jolly"):
    # Predict cluster
    cluster = clf.predict([[income, score]])[0]
    group = cluster_to_group.get(cluster, "Unknown")

    # Predict regression
    model = cluster_regressors.get(cluster)
    if model:
        reg_input = pd.DataFrame([[age, income, genre]], columns=["Age", "Annual Income (k$)", "Genre"])
        result = model.predict(reg_input)[0]
        st.success(f"Cluster: {cluster} ({group})")
        st.success(f"Prediction from Regressor: {result:.2f}")
    else:
        st.error("Regressor not found for this cluster.")

