# train_model.py
import joblib
import pandas as pd

from pipeline import Preprocessor
from models import ModelTrainer

if __name__ == '__main__':
    # Load and preprocess
    preprocessor = Preprocessor('')
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_data('data/flights_train_cleaned.csv')

    # Train model
    trainer = ModelTrainer('')
    trainer.train_linear_regression(X_train, X_test, y_train, y_test)
    trainer.train_xgboost(X_train, X_test, y_train, y_test)
    trainer.train_neural_network(X_train, X_test, y_train, y_test)

    # Load test data
    test_df = pd.read_csv("data/flights_test_cleaned.csv")
    copy_test = test_df.copy()
    X_test_final = Preprocessor('').prepare_test_data(copy_test, f'encoders/onehot_encoder.pkl')

    # Load the best models
    xgb = joblib.load("models/nn_model.joblib")
    preds = xgb.predict(X_test_final)

    # Attach predictions to IDs
    submission = test_df[['id']].copy()
    submission['ARRIVAL_DELAY'] = preds

    # Save to CSV
    submission.to_csv("data/test_predictions.csv", index=False)
    print("✅ Final submission saved to data/test_predictions.csv")

    # lets try the modeling without the outliers and check how the models will perform
    # Load and filter the data for outliers
    train_df = pd.read_csv('data/flights_train_cleaned.csv')

    # Remove outliers
    preprocessor_without_outliers = Preprocessor('_without_outliers')
    filtered_train_df = preprocessor_without_outliers.remove_outliers(train_df, target_col='ARRIVAL_DELAY', lower=0.01, upper=0.99)

    #Save filtered data or continue pipeline
    filtered_train_df.to_csv('data/flights_train_filtered.csv', index=False)
    X_train, X_test, y_train, y_test = preprocessor_without_outliers.prepare_train_data('data/flights_train_filtered.csv')

    #train the models without outliers
    trainer_without_out = ModelTrainer('_without_outliers')
    trainer_without_out.train_linear_regression(X_train, X_test, y_train, y_test)
    trainer_without_out.train_xgboost(X_train, X_test, y_train, y_test)
    trainer_without_out.train_neural_network(X_train, X_test, y_train, y_test)