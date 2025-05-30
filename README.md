# flight-delays-prediction
Machine learning pipeline for predicting US flight arrival delays
Flight Delays Prediction
Predicting US domestic flight arrival delays using machine learning and feature engineering.

# Overview
The goal of this project is to accurately predict flight arrival delays using structured airline and airport data.

Rather than focusing on complex or deep models, the primary objective was to build meaningful features and a robust end-to-end pipeline. This includes detailed data cleaning, data transformation, time features and airport and airline related features.

The modeling approach uses three ml models — Linear Regression, XGBoost, and a simple Neural Network — to compare performance and highlight the impact of feature engineering.

# Project Structure

flight-delays-prediction/

│

├── data/

│   ├── airlines.csv

│   ├── airports.csv

│   ├── flights_train.csv

│   ├── flights_train_cleaned.csv

│   ├── flights_train_filtered.csv

│   ├── flights_test.csv

│   ├── flights_test_cleaned.csv

│   └── test_predictions.csv

│

├── encoders/

│   ├── onehot_encoder.pkl

│   └── onehot_encoder_without_outliers.pkl

│

├── models/

│   ├── linear_model.joblib

│   ├── linear_model_without_outliers.joblib

│   ├── nn_model.joblib

│   ├── nn_model_without_outliers.joblib

│   ├── xgboost_model.joblib

│   └── xgboost_model_without_outliers.joblib

│

├── plots/

│   └── ... (diagnostic plots for all models)

│

├── main.py

├── models.py

├── pipeline.py

├──FlightDelaysPrediction.ipynb

├── .gitignore

├── LICENSE

└── README.md


# Data
This university project with the presented data here: https://www.kaggle.com/c/flight-delays-prediction-challeng2/team
The idea of the project was to build machine learning models based on the data in the flights_train and then to use it to predict the values in flights_test

# Main Features & Engineering
Time-based features: Hour, minute, part of day, day of week, is weekend, season, week of year, and derived delay calculations.

Airport & airline metadata: One-hot encoding of airline and airport codes, categorical time-of-day and weekday/weekend flags.

Outlier handling: Option to remove outliers by filtering ARRIVAL_DELAY to the 1st–99th percentile.

# Modeling Approach
Models trained and compared:

Linear Regression (baseline)

XGBoost Regressor

Feedforward Neural Network (Keras)

Models are trained and evaluated with and without outlier filtering to analyze robustness.

# How to Run
Place data in the data/ directory (flights_train_cleaned.csv and flights_test_cleaned.csv).
Run everything in FlightDelaysPrediction.ipynb to check the visualizations and explanation of the feature engineering.
At the end of the file flights_train_cleaned amd flights_test_cleaned are stored in the data/ directory.
Then Run main.py:

Trains all models on both the full dataset and the outlier-filtered dataset.

Saves metrics, predictions, and diagnostic plots in respective folders.

# Results:

Model metrics (MSE and R²) are printed to console.

Diagnostic plots and test predictions are saved for inspection in plots/

# Outlier Handling
Outliers in the target (ARRIVAL_DELAY) are defined as values below the 1st percentile or above the 99th percentile.
We train and compare all models both with and without these outliers to evaluate the impact of extreme delays on model performance.
