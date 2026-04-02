# Breast Cancer Classification with Machine Learning

A machine learning pipeline to classify breast tumors as **malignant** or **benign** using the Wisconsin Breast Cancer Dataset.

---

## Overview

This project walks through a full data science workflow — from data cleaning and exploration to feature selection and model development — to predict breast cancer diagnoses using cell nucleus measurements.

---

## Dataset

- **Source:** [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Samples:** 569 patients
- **Features:** 30 numerical features (e.g., radius, texture, concavity, symmetry)
- **Target:** Diagnosis — Malignant (1) or Benign (0)

---

## Project Steps

### 1. Data Cleaning
- Removed irrelevant columns (`id`, `Unnamed: 32`)
- Encoded the `diagnosis` column using `pd.get_dummies`
- Dropped the redundant `diagnosis_B` dummy column

### 2. Normalization
- Applied `MinMaxScaler` to scale all features to the range [0, 1]

### 3. Exploratory Data Analysis
- **Class distribution:** ~357 benign vs. ~212 malignant cases
- **Correlation heatmap:** Identified strong multicollinearity among size-related features (radius, perimeter, area)

### 4. Feature Selection
- Manually removed highly correlated features: `perimeter_mean`, `area_mean`, `perimeter_worst`, `radius_worst`, `perimeter_se`
- Used `RandomForestClassifier` to rank remaining features by importance

**Top 5 most important features:**
| Feature | Importance |
|---|---|
| area_worst | 0.180 |
| concave points_mean | 0.177 |
| concave points_worst | 0.146 |
| concavity_mean | 0.090 |
| radius_mean | 0.086 |

### 5. Model Development
- Split data 80/20 into training and test sets using `train_test_split`

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Seaborn & Matplotlib
- Google Colab

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
   ```
2. Open `breastcancer.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook
3. Run all cells from top to bottom

---

## 📊 Results

| Metric | Score |
|---|---|
| Cross-Validation Accuracy (5-fold) | **97%** |
| Optimized Model Accuracy (test set) | **96%** |
| Standard Deviation (CV) | 0.02 |

**Best Hyperparameters** (found via GridSearchCV):
- `n_estimators`: 100
- `max_depth`: None
- `min_samples_split`: 2

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
