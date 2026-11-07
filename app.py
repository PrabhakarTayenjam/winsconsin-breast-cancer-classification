import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Title
st.title("Breast Cancer Diagnosis")

# 1. Model Selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
    ],
)

# Mapping models to the trained models
model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}

# Test data upload
test_data = st.file_uploader("Upload test data", type=["csv"])

if st.button("Evaluate"):
    if test_data is None:
        st.info("Please upload test_data.csv to display evaluation results.")
    else:       
        # Read test dataset
        test_df = pd.read_csv(test_data)
    
        # Separate features and target
        X_test = test_df.drop(columns=["diagnosis"])
        y_test = test_df["diagnosis"]
    
        # Scale the data using the saved scaler
        scaler = joblib.load("model/scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
    
        # Load the selected model
        model = joblib.load(model_files[model_name])
    
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
        # Display the evaluation metrics
        st.subheader(f"Evaluation Metrics: {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
    
        # 4. Confusion Matrix Plot (Requirement 6d)
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)