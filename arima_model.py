"""
ARIMA Model for Solar Irradiance Prediction
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
from path_utils import ensure_dir, X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, ARIMA_PREDICTIONS_FILE, RESULTS_DIR
warnings.filterwarnings('ignore')

def find_optimal_arima_params(timeseries, p_range=(0,5), d_range=(0,2), q_range=(0,5)):
    """
    Find optimal ARIMA parameters using grid search
    """
    print("\n[ARIMA] Finding optimal parameters...")
    best_aic = np.inf
    best_params = None
    
    for p in range(*p_range):
        for d in range(*d_range):
            for q in range(*q_range):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    print(f"✓ Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
    return best_params


def train_arima_model(X_train, X_test, y_train, y_test):
    """
    Train ARIMA model on univariate GHI time series
    
    Note: ARIMA works with univariate data, so we use the target (GHI) sequence
    extracted from X_train
    """
    print("\n" + "="*60)
    print("ARIMA MODEL FOR GHI PREDICTION")
    print("="*60)
    
    # Extract GHI sequences (first column in X_train)
    # Flatten to create univariate time series
    y_train_flat = np.concatenate(X_train[:, :, 0])  # GHI is first feature
    y_test_flat = np.concatenate(X_test[:, :, 0])
    
    print(f"Training time series length: {len(y_train_flat)}")
    print(f"Test time series length: {len(y_test_flat)}")
    
    # Find optimal parameters
    best_params = find_optimal_arima_params(y_train_flat, p_range=(0,4), d_range=(0,2), q_range=(0,4))
    
    # Train ARIMA model
    print(f"\n[ARIMA] Training model with parameters {best_params}...")
    model = ARIMA(y_train_flat, order=best_params)
    arima_results = model.fit()
    
    print(arima_results.summary())
    
    # Make predictions
    print(f"\n[ARIMA] Making predictions...")
    
    # Forecast on test set
    n_periods = len(y_test_flat)
    forecast = arima_results.get_forecast(steps=n_periods)
    y_pred = forecast.predicted_mean.values
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("ARIMA PERFORMANCE METRICS")
    print(f"{'='*60}")
    
    mse = mean_squared_error(y_test_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred)
    r2 = r2_score(y_test_flat, y_pred)
    
    print(f"✓ MSE:  {mse:.6f}")
    print(f"✓ RMSE: {rmse:.6f}")
    print(f"✓ MAE:  {mae:.6f}")
    print(f"✓ R²:   {r2:.6f}")
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.plot(y_test_flat[:500], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500], label='ARIMA Prediction', alpha=0.7, linewidth=2)
    plt.legend(fontsize=12)
    plt.title('ARIMA - GHI Prediction (First 500 samples)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Normalized GHI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ensure_dir(RESULTS_DIR)
    plt.savefig(ARIMA_PREDICTIONS_FILE, dpi=100)
    print(f"\n✓ Plot saved: {ARIMA_PREDICTIONS_FILE}")
    
    return {
        'model': arima_results,
        'y_pred': y_pred,
        'y_test': y_test_flat,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(X_TRAIN_FILE)
    X_test = np.load(X_TEST_FILE)
    y_train = np.load(Y_TRAIN_FILE)
    y_test = np.load(Y_TEST_FILE)
    
    # Create output directory
    ensure_dir(RESULTS_DIR)
    
    # Train model
    results = train_arima_model(X_train, X_test, y_train, y_test)
    
    print(f"\n✅ ARIMA TRAINING COMPLETE!")
