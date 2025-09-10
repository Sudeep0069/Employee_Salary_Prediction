# Employee_Salary_Prediction
🪙 Employee Salary Prediction App
This repository contains a machine learning project focused on predicting employee income levels based on various demographic and employment features. The project includes a Jupyter Notebook for data exploration and model training, and a Streamlit application for interactive predictions.

🌟 Project Overview
The main goal of this project is to build a classification model that predicts whether an employee's annual income is greater than $50K or less than or equal to $50K. This prediction is based on attributes such as age, education, occupation, hours per week, and years of experience. The Streamlit app provides a user-friendly interface to test the trained model with custom inputs.

📁 Repository Contents
Employee_sal_prediction.ipynb: This Jupyter Notebook contains the complete workflow of the project, including:

Data loading and initial exploration.

Data preprocessing (e.g., handling categorical features, scaling numerical data).

Exploratory Data Analysis (EDA) and visualizations.

Model training (likely using a GradientBoostingClassifier as indicated by the best_model.pkl snippet).

Model evaluation and selection.

Saving the trained model.

emp_sal_prediction.csv: The dataset used for training and evaluating the salary prediction model. It contains various features related to employees and their income levels.

best_model.pkl: A serialized (pickled) file containing the trained machine learning model. This model is loaded by the Streamlit application for making predictions.

app.py: The Python script for the Streamlit web application. This app allows users to input employee details through a simple interface and get an instant salary prediction (whether income is >50K or <=50K).

🛠️ Technologies and Libraries
The project leverages the following key Python libraries:

pandas: For data manipulation and analysis.

numpy: For numerical operations.

scikit-learn: For machine learning model training (e.g., GradientBoostingClassifier), preprocessing, and evaluation.

joblib: For saving and loading the machine learning model.

streamlit: For creating the interactive web application.
