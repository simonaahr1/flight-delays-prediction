import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy import sparse
import joblib
import os
from scipy.sparse import hstack

class Preprocessor:
    """
    Handles all preprocessing steps for model training and inference,
    including time conversion, encoding categorical variables, and feature assembly.
    """
    def __init__(self, outliers):
        # List of categorical columns to be one-hot encoded
        self.categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TIME_OF_DAY', 'DAY_OF_WEEK']
        # OneHotEncoder to transform categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        self.outliers = outliers

    def prepare_train_data(self, filepath):
        """
        Loads training data, encodes categorical features, combines all features into a single sparse matrix,
        and splits data into train/test sets.
        """
        df = pd.read_csv(filepath)
        # Convert time columns (in HH:MM:SS) to elapsed minutes for model compatibility
        for col in ['SCHEDULED_DEPARTURE_TIME', 'DEPARTURE_TIME', 'WHEELS_OFF', 'SCHEDULED_ARRIVAL']:
            df[col] = df[col].apply(self.time_to_minutes)

        # Separate categorical and numerical features
        X_cat = df[self.categorical_cols]
        X_num = df.drop(columns=self.categorical_cols + ['ARRIVAL_DELAY'])
        y = df['ARRIVAL_DELAY']

        # Fit the encoder on training categorical data and save it for future use
        self.encoder.fit(X_cat)
        os.makedirs('encoders', exist_ok=True)
        joblib.dump(self.encoder, f'encoders/onehot_encoder{self.outliers}.pkl')

        # Transform categorical data and stack it with numerical data into a single feature matrix
        X_cat_encoded = self.encoder.transform(X_cat)
        X_final = hstack([X_cat_encoded, sparse.csr_matrix(X_num.astype(np.float32).values)])
        return train_test_split(X_final, y, test_size=0.33, random_state=42)

    @staticmethod
    def time_to_minutes(t):
        """
        Converts a time string 'HH:MM:SS' to elapsed minutes since midnight.
        If already numeric or NaN, returns as is.
        """
        if isinstance(t, str):
            h, m, s = map(int, t.split(':'))
            return h * 60 + m + s / 60
        return t  # in case it's already a float or nan

    def prepare_test_data(self, df, encoder_path, categorical_cols=None):
        """
        Processes and encodes the test set using the previously fitted encoder,
        returning a sparse feature matrix ready for prediction.
        """
        if categorical_cols is None:
            categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TIME_OF_DAY', 'DAY_OF_WEEK']

        # Convert time columns to elapsed minutes
        for col in ['SCHEDULED_DEPARTURE_TIME', 'DEPARTURE_TIME', 'WHEELS_OFF', 'SCHEDULED_ARRIVAL']:
            df[col] = df[col].apply(Preprocessor.time_to_minutes)

        # Load the saved one-hot encoder
        encoder = joblib.load(encoder_path)
        X_cat = df[categorical_cols]
        X_cat_encoded = encoder.transform(X_cat)

        # Combine categorical and numerical features into final input matrix
        X_num = df.drop(columns=categorical_cols + ['id'])
        X_final = hstack([X_cat_encoded, sparse.csr_matrix(X_num.astype(np.float32).values)])

        return X_final

    @staticmethod
    def remove_outliers(df, target_col, lower=0.01, upper=0.99):
        """
        Removes rows in df where target_col is outside the [lower, upper] percentiles.
        Returns a filtered DataFrame.
        """
        low = df[target_col].quantile(lower)
        high = df[target_col].quantile(upper)
        filtered_df = df[(df[target_col] >= low) & (df[target_col] <= high)]
        print(f"Rows before: {len(df)}, after filtering: {len(filtered_df)}")
        return filtered_df



