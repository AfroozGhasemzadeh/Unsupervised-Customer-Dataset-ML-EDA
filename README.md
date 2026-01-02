# Customer Segmentation Using Unsupervised Learning
This project applies a range of unsupervised machine learning techniques to segment credit card customers based on their financial behavior. The dataset contains sixteen numerical features describing balances, purchase patterns, cash advances, credit limits, payments, and account tenure. Since no labels are provided, the goal is to uncover natural groupings within the customer base and derive meaningful insights that can support targeted marketing, risk assessment, and customer lifecycle strategies.
The workflow is organized into three structured notebooks—Data Processing, Exploratory Data Analysis (EDA), and Modelling—with all generated figures saved in dedicated folders for clarity and reproducibility.
# Dataset
The dataset used in this project comes from the Customer Data dataset published on Kaggle by mbsoroush: 
https://www.kaggle.com/datasets/mbsoroush/customer-data
It includes 8,950 anonymized credit card customers and 16 numerical features, such as:
- BALANCE
- PURCHASES & PURCHASES_FREQUENCY
- ONEOFF_PURCHASES
- INSTALLMENTS_PURCHASES
- CASH_ADVANCE & CASH_ADVANCE_TRX
- CREDIT_LIMIT
- PAYMENTS
- PRC_FULL_PAYMENT
- TENURE
The dataset is fully numerical, making it ideal for clustering and dimensionality reduction techniques.
# Notebooks

# Data Processing Notebook
This notebook focuses on preparing the dataset for modelling:
- Handling missing values (median imputation, inf → NaN replacement)
- Removing outliers where appropriate
- Scaling features using StandardScaler
- Saving the cleaned dataset for downstream notebooks
- Generating and saving preprocessing figures in
Customer Dataset Figures/Data Processing/
This ensures a clean, standardized input for all clustering algorithms.
# Exploratory Data Analysis (EDA) Notebook
The EDA notebook investigates the structure and distribution of the data:
- Feature distributions and KDE plots
- Correlation heatmaps
- PCA for dimensionality understanding
- Summary statistics and behavioral patterns
All EDA figures are saved in: Customer Dataset Figures/Data Processing/
This step provides intuition about the dataset and helps guide modelling decisions.
# Modelling Notebook
The modelling notebook evaluates multiple unsupervised learning algorithms to identify meaningful customer segments.
The following clustering methods were explored:
Clustering Algorithms
- K‑Means
- MiniBatch K‑Means
- MeanShift
- DBSCAN
- Gaussian Mixture Models (GMM)
- Agglomerative Hierarchical Clustering
- Dendrogram analysis (Ward linkage)
Model Evaluation Metrics
- Inertia (Elbow Method)
- Silhouette Score
- Calinski–Harabasz Index
- PCA‑based cluster visualization
- KDE distributions per cluster
# Final Model
To keep the notebook efficient and lightweight, the final segmentation model is based on K‑Means, selected for its stability, interpretability, and strong performance across evaluation metrics.
All modelling figures are saved in: Dataset Figures/Modelling/
# Project Goal
The objective of this project is to identify natural customer segments that can support:
- targeted marketing strategies
- personalized product offerings
- credit risk profiling
- customer retention and lifecycle management
By comparing multiple clustering algorithms and selecting the most effective model, the project demonstrates a complete unsupervised learning workflow—from raw data to actionable segmentation.



