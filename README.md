# 📚 Student Performance Prediction Model

**Author:** Eng. Abdallah Dwikat | **Date:** 17th September 2025

## 📝 Project Overview

This project involved building a sophisticated **regression model** to predict student exam scores based on a wide range of academic, lifestyle, and environmental factors. The goal was to identify the key drivers of student performance and create a highly accurate predictive tool using **Kaggle data** and advanced **machine learning pipelines**.

## 💻 Technical Skills Demonstrated

*   **Data Handling:** `Pandas`, `NumPy`, Data Imputation, Feature Encoding (`OrdinalEncoder`, `OneHotEncoder`)
*   **Advanced Preprocessing:** `Scikit-learn Pipelines`, `ColumnTransformer`, `PolynomialFeatures`
*   **Machine Learning Models:** `Linear Regression`, `RidgeCV`, `LassoCV`
*   **Model Evaluation:** `MAE`, `RMSE`, `R² Score`, Residual Analysis, Cross-Validation
*   **Data Visualization:** `Matplotlib`, `Seaborn` for correlation heatmaps and insightful plots
*   **Reproducibility:** Setting random seeds, version control with Kaggle datasets via `kagglehub`

## 🚀 Process & Methodology

### 1. **Data Acquisition & Initial Inspection**
*   Downloaded the latest version of the **"Student Performance Factors"** dataset directly from Kaggle using the `kagglehub` API.
*   Performed initial inspection on a dataset of **6,607 students** with **20 features**, identifying data types and detecting missing values (~1% in a few columns).

### 2. **Exploratory Data Analysis (EDA) & Visualization**
*   Conducted a **correlation analysis** on numeric features, revealing that `Attendance` and `Hours_Studied` were the strongest predictors of `Exam_Score`.
*   Created **boxplots** and **scatter plots** to visualize the relationship between categorical features (e.g., `Teacher_Quality`) and the target score.

### 3. **Advanced Data Preprocessing Pipeline**
*   Engineered a robust preprocessing pipeline using `ColumnTransformer` to handle different data types separately:
    *   **Numeric Features:** Imputed with median and applied `PolynomialFeatures` to `Hours_Studied` and `Attendance`.
    *   **Ordinal Features:** (e.g., `Motivation_Level`, `Distance_from_Home`) Imputed with mode and encoded using `OrdinalEncoder` with specified category orders.
    *   **Nominal Features:** (e.g., `Gender`, `Internet_Access`) Imputed with mode and encoded using `OneHotEncoder`.

### 4. **Model Building, Training & Regularization**
*   Established a **baseline model** (predicting the mean) for performance comparison.
*   Trained and evaluated multiple regression models:
    *   **Linear Regression**
    *   **Polynomial Regression** (degree=2)
    *   **Regularized Models:** `RidgeCV` and `LassoCV` with cross-validation to find the optimal regularization strength and prevent overfitting.
*   Performed **residual analysis** to validate model assumptions and ensure prediction quality.

## 📊 Results & Performance

The final **regularized linear model** demonstrated exceptional predictive power, significantly outperforming the baseline.

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Mean)** | 2.82 | 3.76 | 0.00 |
| **Linear Regression** | **0.45** | **3.24** | **0.771** |
| **Polynomial Regression** | **0.45** | **3.24** | **0.771** |
| **Ridge Regression** | **0.45** | **3.24** | **0.771** |
| **Lasso Regression** | **0.45** | **3.24** | **0.771** |

*   The **low MAE (0.45)** indicates the model's predictions are, on average, less than half a point away from the actual exam scores.
*   The **high R² score (0.771)** shows the model explains over 77% of the variance in the exam scores, confirming its effectiveness.

## 🔑 Key Insights

The analysis provided clear, actionable insights into the factors affecting student performance:
*   **Strongest Predictors:** `Attendance` and `Hours_Studied` showed the highest positive correlation with exam scores.
*   **Diminishing Returns:** The relationship between `Hours_Studied` and `Exam_Score` was found to be non-linear, best captured with polynomial features, indicating potential diminishing returns after a certain point.
*   **Minimal Regularization Needed:** The best alpha for Lasso was very small (`0.001`), suggesting that the well-prepared features were already robust and that strong regularization was not necessary to prevent overfitting.

---
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Kaggle API
