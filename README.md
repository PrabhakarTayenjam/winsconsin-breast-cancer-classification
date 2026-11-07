# Breast Cancer Wisconsin Classification

## a. Problem Statement
Classification of breast cancer as **malignant (1)** or **benign (0)**

---

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / Kaggle
- **Total Instances:** 569 instances
- **Number of Features:** 30 numerical features (excluding target `diagnosis` and `id`)
- **Target Variable:** `diagnosis` (`M` = Malignant mapped to `1`, `B` = Benign mapped to `0`)
- **Train/Test Split:** 80% Training (455 instances), 20% Testing (114 instances)

---

## c. GitHub Repository Link
- **GitHub Repository:** [Insert Your GitHub Repository Link Here]
- **Live Streamlit App:** [Insert Your Deployed Streamlit App Link Here]

---

## d. Models Used & Comparison Table

### 1. Performance Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9825 | 0.9974 | 0.9762 | 0.9762 | 0.9762 | 0.9619 |
| Decision Tree | 0.9386 | 0.9358 | 0.9268 | 0.9048 | 0.9157 | 0.8668 |
| KNN | 0.9561 | 0.9859 | 0.9744 | 0.9048 | 0.9383 | 0.9046 |
| Naive Bayes | 0.9386 | 0.9881 | 0.9487 | 0.8810 | 0.9136 | 0.8666 |
| Support Vector Machine | 0.9737 | 0.9974 | 0.9756 | 0.9524 | 0.9639 | 0.9429 |

---

### 2. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved high overall performance in all metrics |
| **Decision Tree** | Low performance in all metrics compared to other models |
| **KNN** | High permormance in other metrics but low recall value (0.9048) |
| **Naive Bayes** | Low performance and lowest recall value (0.8810) |
| **Support Vector Machine** | High performace in all metrics but recall is lower than Logistic Regression |
| **Overall Winner for your dataset?** | **Logistic Regression** is the overall winner for this dataset, achieving top performance in all metrics and the **highest recall vallue (0.9762)**. High recall is particularly critical in medical classification to minimize false negatives. |