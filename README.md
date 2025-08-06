# Credit-Card-Fraud-Detection-using-Machine-Learning
# 🔐 Credit Card Fraud Detection using Machine Learning

A machine learning-based system designed to detect fraudulent transactions using advanced classification models. This project evaluates multiple ML algorithms and identifies the best-performing model (XGBoost) based on key performance metrics like F1-Score, Recall, AUC-ROC, and more.

---

## 🚀 Project Overview

This project aims to build a robust Intrusion Detection System (IDS) to detect fraudulent credit card transactions. By applying multiple machine learning models, the goal is to accurately classify legitimate and fraudulent behavior in transactional data, which is often highly imbalanced.

---

## 🧠 ML Pipeline

- **Data Cleaning**: Removed irrelevant fields (`nameOrig`, `nameDest`), handled missing values.
- **Feature Engineering**:
  - Used `OneHotEncoding` for categorical variables
  - Applied `StandardScaler` to numerical features
- **Imbalanced Handling**: Used **SMOTE** to oversample the minority class (fraud)
- **Model Training**: Trained and evaluated 5 different classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost (Gradient Boosting)
  - Naive Bayes
- **Metric Evaluation**: Evaluated using:
  - Accuracy
  - Precision (Fraud)
  - Recall (Fraud)
  - F1-Score (Fraud)
  - AUC-ROC

---

## 🏆 Best Model: XGBoost

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | 0.99   |
| Precision (Fraud)   | 0.89   |
| Recall (Fraud)      | 0.88   |
| F1-Score (Fraud)    | 0.88   |
| AUC-ROC             | 0.98   |

✅ XGBoost achieved the highest performance across all key fraud-detection metrics, making it the most reliable model in this pipeline.

---

## 📦 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-red?logo=xgboost)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-black?logo=pandas)
![Colab](https://img.shields.io/badge/GoogleColab-Notebook-yellow?logo=googlecolab)

---

## 📈 Key Highlights

- 🚩 Handled extreme class imbalance using SMOTE
- 🔍 Compared 5 ML models with detailed metrics
- 🎯 Final model selection based on F1, Recall, and AUC — not just accuracy
- 📊 Normalized scoring system to choose best model across all metrics

---

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python credit_card_fraud_detection.py
   ```

---

## 📁 Dataset

- 📄 **File**: `transactions.csv`  
- 🔢 ~21,000 Modbus transactions  
- 🧾 Columns: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `type`, etc.  
- 🏷️ Target: `isFraud` (0 = normal, 1 = fraud)

The dataset used in this project is a simulated Modbus transaction log stored as `transactions.csv`. It includes both normal and fraud-classified transactions.

---



## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.
