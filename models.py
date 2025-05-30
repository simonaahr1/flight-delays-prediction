from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
import os
import seaborn as sns


class ModelTrainer:
    """
   Class for training, evaluation, and plot for multiple regression models
   (Linear Regression, XGBoost, and a Neural Network).
   """
    def __init__(self, outliers):
        # Initialize models for later training
        self.linear_model = LinearRegression()
        self.xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=1)
        self.nn_model = None  # will be initialized in train_neural_network
        self.outliers = outliers

    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """
        Trains a Linear Regression model, evaluates performance, print the diagrams, and saves results.
        """
        print(f"\nTraining Linear Regression {self.outliers}...")
        self.linear_model.fit(X_train, y_train)
        preds = self.linear_model.predict(X_test)
        self._print_metrics(y_test, preds, "Linear Regression")
        self._plot_diagnostics(y_test, preds, "Linear Regression")
        dump(self.linear_model, f'models/linear_model{self.outliers}.joblib')

    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """
        Trains a XGBoost model, evaluates performance, print the diagrams, and saves results.
        """
        print(f"\nTraining XGBoost {self.outliers}...")
        self.xgb_model.fit(X_train, y_train)
        preds = self.xgb_model.predict(X_test)
        self._print_metrics(y_test, preds, "XGBoost")
        self._plot_diagnostics(y_test, preds, "XGBoost")
        dump(self.xgb_model, f'models/xgboost_model{self.outliers}.joblib')


    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """
        Builds and trains a feedforward neural network, applies early stopping,
        evaluates performance, and saves results.
        """
        print(f"\nTraining Neural Network {self.outliers}...")
        self.nn_model = self._build_neural_network(X_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.nn_model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stop]
        )
        preds = self.nn_model.predict(X_test).flatten()
        self._print_metrics(y_test, preds, "Neural Network")
        self._plot_diagnostics(y_test, preds, "Neural Network")
        dump(self.nn_model, f'models/nn_model{self.outliers}.joblib')


    def _build_neural_network(self, input_dim):
        """
        Internal helper to construct a simple dense neural network for regression.
        """
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _print_metrics(self, y_true, y_pred, model_name):
        """
        Calculates and prints key regression metrics for the given model.
        We use MSE and R²
        MSE measures the average squared difference between your model’s predicted values and the actual values.
        R² shows how well your model’s predictions explain the variance in the actual target values.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} Results → MSE: {mse:.2f} | R²: {r2:.4f}")

    def _plot_diagnostics(self, y_true, y_pred, model_name):
        """
        Creates and saves diagnostic plots:
        - Actual vs Predicted values of the models scatter plot
        - Residual plot
        """
        os.makedirs("plots", exist_ok=True)

        # Actual vs Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted - {model_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_actual_vs_predicted{self.outliers}.png")
        plt.close()

        # Residual Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 5))
        sns.residplot(x=y_pred, y=residuals, scatter_kws={'alpha': 0.5})
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"Residual Plot - {model_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_residual_plot{self.outliers}.png")
        plt.close()
