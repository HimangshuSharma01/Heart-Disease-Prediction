# Heart Disease Prediction Workflow and Streamlit App

This project focuses on predicting the presence of heart disease using a dataset of clinical features. Various machine learning models are evaluated, and the best-performing model (K-Nearest Neighbors) is saved alongside its preprocessing assets to power a Streamlit web application.

## Project Overview

Heart disease is a major global health concern. Early detection and automated risk assessment can assist medical professionals in making timely decisions. This repository contains the end-to-end workflow:
1. Data Exploration and Profiling: Analyzing data patterns using automated profiling tools.
2. Data Cleaning: Handling logical inconsistencies and clinical outliers (e.g., correcting zero-value blood pressure and cholesterol levels).
3. Feature Engineering: One-hot encoding categorical features and standardizing numerical scales.
4. Model Training and Evaluation: Training classifiers including Logistic Regression, KNN, Naive Bayes, Decision Trees, and Support Vector Machines.
5. Deployment Preparation: Exporting the trained model, scaler, and column configurations for runtime predictions.
6. Web Application: A Streamlit application (app.py) for real-time user predictions.

## Dataset

The model is trained on the Heart Failure Prediction Dataset (heart.csv), which contains 11 clinical features used to predict heart disease:
- Age: Age of the patient
- Sex: Sex of the patient (M/F)
- ChestPainType: Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- RestingBP: Resting blood pressure (mm Hg)
- Cholesterol: Serum cholesterol (mm/dl)
- FastingBS: Fasting blood sugar (1 if FastingBS > 120 mg/dl, 0 otherwise)
- RestingECG: Resting electrocardiogram results (Normal, ST, LVH)
- MaxHR: Maximum heart rate achieved (71-202)
- ExerciseAngina: Exercise-induced angina (Y/N)
- Oldpeak: ST depression induced by exercise relative to rest
- ST_Slope: The slope of the peak exercise ST segment (Up, Flat, Down)
- HeartDisease: Output class (1: heart disease, 0: normal)

## Data Preprocessing Details

- Logical Corrections: Replaced clinical anomalies, such as resting blood pressure and cholesterol values of 0, with their respective column means (excluding zeros).
- Encoding: Categorical features were converted to dummy variables using one-hot encoding.
- Scaling: Numerical features ('Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak') were scaled using StandardScaler to ensure optimal performance for distance-based models like KNN.

## Model Performance

After splitting the dataset into 80% training and 20% testing sets, the models achieved the following accuracy and F1 scores on the test set:

| Model | Accuracy | F1 Score |
| :--- | :---: | :---: |
| K-Nearest Neighbors (KNN) | 89.13% | 90.29% |
| Logistic Regression | 89.13% | 90.29% |
| Naive Bayes | 87.50% | 88.44% |
| SVM (RBF Kernel) | 85.87% | 87.50% |
| Decision Tree | 76.63% | 78.82% |

Both KNN and Logistic Regression achieved the highest performance. The KNN model was selected and exported.

## Saved Artifacts

To ensure consistent preprocessing and predictions inside the Streamlit app, the following artifacts were exported using joblib:
- knn_heart.pkl: The trained K-Nearest Neighbors classifier.
- scaler.pkl: The fitted StandardScaler object to normalize incoming real-time inputs.
- cols.pkl: A serialized list of feature columns to maintain the precise structure and sequence of dummy variables.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-name>
   ```

2. Install required packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
