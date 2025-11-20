# ❤️ Heart Disease Prediction with KNN
---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange)
![Learning](https://img.shields.io/badge/Purpose-Learning-yellow)

KNN Machine Learning Model | Data Preprocessing | EDA | Hyperparameter Tuning

---

## 📘 Overview  

This project demonstrates a **Heart Disease Prediction system** using **K-Nearest Neighbors (KNN)**.  
It includes **complete preprocessing**, **handling missing values**, **outlier removal**, **feature scaling**, **encoding**, and **model evaluation**.  

The notebook is intended for **learning purposes**, and the trained model can later be deployed in a **Flask web application** with a user-friendly HTML + Bootstrap interface for real-time predictions.

---

## 🚀 Key Features  

### 🧪 Data Preprocessing
- Handle missing values for numerical and categorical features
- Outlier detection and removal
- Remove duplicates
- Balance classes (oversampling)
- Feature scaling: MinMaxScaler / StandardScaler
- Encode binary, ordinal, and nominal columns

### 📊 Exploratory Data Analysis (EDA)
- Feature distributions and correlations
- Heatmaps for missing values
- Visualization before and after encoding

### 🤖 KNN Model
- Default KNN model training with cross-validation
- Hyperparameter tuning using `GridSearchCV`
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Compare default vs tuned KNN model

### 💾 Saved Artifacts
- `preprocessor.pkl` → preprocessing pipeline
- `knn_heart_disease_model.pkl` → default trained model
- `knn_tuned_model.pkl` → tuned KNN model

---

## 🧰 Tech Stack  

| Category | Technologies Used |
|-----------|-------------------|
| 💻 Language | Python 3.8+ |
| 🗄️ Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, joblib |
| 🧠 ML Model | K-Nearest Neighbors (KNN) |
| 📊 Visualization | Matplotlib, Seaborn |
| ⚙️ IDE | Jupyter Notebook |
| 📂 Data | CSV (`heart_disease.csv`) |
| 🧰 Deployment | Flask + HTML + Bootstrap (future work) |

---

## 🏗️ Project Structure

## 📂 Dataset

- **Source:** [Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)  
- **Target Variable:** Heart Disease Status  
- **Features:** Numerical, Binary, Ordinal, Nominal (as described in notebook)

---

## ⚡ Notes

- Notebook designed for **learning purposes**.  
- Preprocessing steps are clearly documented for **educational understanding**.  
- Future work: integrate into a **Flask + Bootstrap web interface** for real-time predictions.

---

## 👨‍💻 Author

**Sisuru Disnaka** - 🌐 [GitHub Profile](https://github.com/SisuruDisnaka) 
<br>📧 <a href="mailto:sisurudisnaka001@gmail.com">Contact me</a>

---

## 📜 License

Licensed under the **MIT License** - free for educational and personal use.  
Modify, reuse, and distribute **with attribution**.


