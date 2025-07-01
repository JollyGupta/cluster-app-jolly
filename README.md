# Mall Customer Segmentation & Prediction App

**Built an end-to-end ML solution for mall customer analysis.**  
Used **KMeans** and **Hierarchical Clustering** for segmentation, evaluated with **Silhouette Score**, **Adjusted Rand Score**, and **Mutual Information Score**.  
Trained a **Random Forest Classifier** to predict customer segments and applied **regression models** to estimate **spending scores**.  
Saved models as `.pkl` files. Developed **three Streamlit apps** (1 classification, 2 regression) and deployed on **Streamlit Cloud**.

---

###  App Development & Deployment  
**Tools**: `Anaconda Command Prompt`, `Streamlit`, `Streamlit Cloud`

- **Anaconda Command Prompt**: Used to run development commands like `streamlit run app.py`, manage environments, and install dependencies.
- **Streamlit**: Built interactive ML web apps.
- **Streamlit Cloud**: Deployed apps for public access.

---

### Libraries  
**pandas**, **seaborn**, **matplotlib**, **sweetviz**, **ydata-profiling**, **yellowbrick**, **scikit-learn**, **joblib**
### Libraries Used

- **pandas** – For data manipulation and preprocessing (e.g., cleaning, filtering, grouping)
- **seaborn** – For advanced data visualization, including categorical plots and heatmaps
- **matplotlib** – For custom and static visualizations like bar charts, line plots, and scatter plots
- **sweetviz** – For automated EDA (Exploratory Data Analysis) with comparison reports and feature insights
- **ydata-profiling** – Generates detailed profiling reports of datasets (previously called pandas-profiling)
- **yellowbrick** – For model visualization and performance diagnostics like Elbow plots, Residuals, Feature Importance, etc.
- **scikit-learn** – Core ML library used for clustering, classification, regression, metrics, and model evaluation
- **joblib** – For saving and loading ML models as `.pkl` files (used in deployment)


---

### ML Algorithms  

- **Clustering** – `KMeans`, `KMeans++ Init`, `Hierarchical (Agglomerative)`, `DBSCAN`  
- **Classification** – `Random Forest Classifier`  
- **Regression** – `Linear Regression`, `Random Forest Regressor`

  ### Machine Learning Algorithms Used

- **Clustering**:
  - `KMeans`: Segmented customers using `Annual Income (k$)` and `Spending Score (1–100)` to identify distinct customer groups such as **young high spenders**, **cautious wealthy**, etc.
  - `KMeans++ Init`: Used to initialize centroids more effectively in KMeans, leading to faster convergence and better-defined clusters.
  - `Hierarchical (Agglomerative)`: Helped visualize the structure of customer similarity using dendrograms and supported identifying the optimal number of clusters.
  - `DBSCAN`: Identified clusters based on density and detected outlier customers without requiring a pre-defined number of clusters.

- **Classification**:
  - `Random Forest Classifier`: Trained on customer features like `Age`, `Gender`, and `Annual Income` to **predict the cluster (segment)** a new customer would most likely belong to, based on previously labeled clustering results.

- **Regression**:
  - `Linear Regression`: Estimated a customer's `Spending Score` using basic features such as `Age`, `Gender`, and `Annual Income`.
  - `Random Forest Regressor`: Modeled complex, non-linear relationships to predict `Spending Score` more accurately using the same input features.



![Mall_Customer](https://github.com/user-attachments/assets/5f71c47d-c190-4497-9d55-83d35e8dd9e7)


