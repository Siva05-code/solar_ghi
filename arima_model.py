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
    
    # Handle multi-site data - use y_train/y_test directly (targets, not features)
    # For multi-site: y_train shape is (n_samples, n_sites), take first site or average
    if len(y_train.shape) > 1:
        print(f"[ARIMA] Multi-site data detected: y_train shape {y_train.shape}")
        print("[ARIMA] Using average across sites for univariate ARIMA")
        y_train_flat = np.nanmean(y_train, axis=1)  # Average across sites
        y_test_flat = np.nanmean(y_test, axis=1)
    else:
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
    
    # Remove any NaN or inf values
    mask_train = ~(np.isnan(y_train_flat) | np.isinf(y_train_flat))
    y_train_flat = y_train_flat[mask_train]
    
    mask_test = ~(np.isnan(y_test_flat) | np.isinf(y_test_flat))
    y_test_flat = y_test_flat[mask_test]
    
    print(f"Training time series length: {len(y_train_flat)}")
    print(f"Test time series length: {len(y_test_flat)}")
    
    # Find optimal parameters
    best_params = find_optimal_arima_params(y_train_flat, p_range=(0,4), d_range=(0,2), q_range=(0,4))
    
    # Fallback to default if no parameters found
    if best_params is None:
        print("[ARIMA] ⚠ No optimal parameters found, using default (1,1,1)")
        best_params = (1, 1, 1)
    
    # Train ARIMA model
    print(f"\n[ARIMA] Training model with parameters {best_params}...")
    try:
        model = ARIMA(y_train_flat, order=best_params)
        arima_results = model.fit()
    except Exception as e:
        print(f"[ARIMA] ⚠ Failed to fit with {best_params}, trying (1,1,1): {str(e)[:50]}")
        best_params = (1, 1, 1)
        model = ARIMA(y_train_flat, order=best_params)
        arima_results = model.fit()
    
    print(arima_results.summary())
    
    # Make predictions
    print(f"\n[ARIMA] Making predictions...")
    
    # Forecast on test set
    n_periods = len(y_test_flat)
    forecast = arima_results.get_forecast(steps=n_periods)
    # Handle both pandas Series and numpy array returns from different statsmodels versions
    y_pred = forecast.predicted_mean
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    else:
        y_pred = np.asarray(y_pred)
    
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
    
    # For multi-site output format, repeat predictions for each site
    # This maintains compatibility with orchestrator expectations
    if len(y_test.shape) > 1:
        num_sites = y_test.shape[1]
        y_pred_multisite = np.tile(y_pred[:, np.newaxis], (1, num_sites))
        print(f"\n[ARIMA] Expanding predictions to {num_sites} sites")
    else:
        y_pred_multisite = y_pred
    
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
        'y_pred': y_pred_multisite,
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
