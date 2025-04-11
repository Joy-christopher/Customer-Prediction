
# Predicting Customer Behaviour with Machine Learning for Small Businesses

## 📘 Project Overview

This repository contains the code, data processing steps, and results of a dissertation project that explores how machine learning can be used to predict customer churn, particularly in the banking sector, to support small businesses in enhancing their customer retention and engagement strategies.

**Author**: Joy Nna Christopher  
**Degree**: MSc Data Science  
**Institution**: School of Engineering, Computing & Mathematical Sciences  
**Supervisors**: Dr. Andrew Gascoyne, Pooja Kaur  
**Student Number**: 2305576

---

## 🎯 Objective

The core aim of this project is to build predictive analytics models that small and medium-sized enterprises (SMEs) can use to:

- Identify key customer behaviour metrics
- Predict customer churn with high accuracy
- Offer practical recommendations for CRM integration
- Enhance customer engagement and marketing efforts

---

## 🧠 Technologies Used

- **Python** (Jupyter Notebook)
- Libraries:
  - `pandas`, `numpy` (data handling)
  - `matplotlib`, `seaborn` (data visualization)
  - `scikit-learn` (ML models)
  - `imblearn` (resampling techniques like SMOTE)

---

## 📊 Dataset

- **Name**: `Churn_Modelling`
- **Source**: [Kaggle](https://www.kaggle.com/)
- Contains 10,000 bank customer records with demographic, financial, and behavioral features.
- Target variable: `Exited` (1 = churned, 0 = retained)

---

## 🧪 Methodology

- **Data Preprocessing**:
  - Handling class imbalance using:
    - Undersampling
    - Oversampling
    - SMOTE (Synthetic Minority Over-sampling Technique)
  - Feature scaling and encoding
  - Train/Test split (80/20)

- **Machine Learning Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## ✅ Results Summary

- **Best Performing Models**:
  - **Random Forest**: Achieved **94% accuracy** with SMOTE
  - **Gradient Boosting**: Also performed well across all metrics
- Balanced precision, recall, and F1-scores make these models effective for churn prediction.

---

## 💡 Key Findings

- Demographics, engagement, and financial data are crucial in predicting churn.
- SMOTE significantly enhances model performance on imbalanced datasets.
- Small businesses can adopt these techniques using open-source tools and cloud-based platforms.

---

## 📌 Limitations

- Dataset is specific to the banking sector
- Traditional ML models only; deep learning or real-time systems not explored

---

## 🔭 Future Work

- Expand study with diverse industry datasets
- Explore advanced ML and real-time prediction
- Integrate models into CRM systems

---

## 📁 Project Structure

```
.
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── models/                 # Saved ML models (optional)
├── figures/                # Visualizations (confusion matrices, feature importance, etc.)
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
```

---

## 📜 License

This project is for academic use. Refer to the original Kaggle dataset license for data usage permissions.

---

## 🙌 Acknowledgements

Thanks to Kaggle for providing the dataset and to my supervisors for their guidance throughout this project.
