import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import os
from path_utils import X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, RESULTS_DIR, ensure_dir
warnings.filterwarnings('ignore')

def reshape_sequences_for_ml(X, seq_len=24):
    """
    Reshape 3D sequences (n_samples, seq_len, n_features)
    to 2D features (n_samples, seq_len * n_features)
    for Random Forest and XGBoost
    """
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(n_samples, seq_len * n_features)
    print(f"✓ Reshaped from {X.shape} to {X_reshaped.shape}")
    return X_reshaped


def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=20):
    """
    Train Random Forest model
    """
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL FOR GHI PREDICTION")
    print("="*60)

    # Reshape sequences
    X_train_2d = reshape_sequences_for_ml(X_train)
    X_test_2d = reshape_sequences_for_ml(X_test)

    print(f"\n[RF] Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf_model.fit(X_train_2d, y_train)

    # Make predictions
    print(f"\n[RF] Making predictions...")
    y_pred = rf_model.predict(X_test_2d)

    # Calculate metrics
    print(f"\n{'='*60}")
    print("RANDOM FOREST PERFORMANCE METRICS")
    print(f"{'='*60}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"✓ MSE:   {mse:.6f}")
    print(f"✓ RMSE:  {rmse:.6f}")
    print(f"✓ MAE:   {mae:.6f}")
    print(f"✓ R²:    {r2:.6f}")
    print(f"✓ MAPE:  {mape:.6f}")

    # Feature importance
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
    print(f"\n✓ Top 10 most important features (indices): {top_features_idx}")

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:500], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500], label='RF Prediction', alpha=0.7, linewidth=2)
    plt.legend(fontsize=12)
    plt.title('Random Forest - GHI Prediction (First 500 samples)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Normalized GHI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/random_forest_predictions.png', dpi=100)
    print(f"\n✓ Plot saved: {RESULTS_DIR}/random_forest_predictions.png")

    return {
        'model': rf_model,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }


def train_xgboost(X_train, X_test, y_train, y_test, max_depth=7, learning_rate=0.1, n_estimators=200):
    """
    Train XGBoost model
    """
    print("\n" + "="*60)
    print("XGBOOST MODEL FOR GHI PREDICTION")
    print("="*60)

    # Reshape sequences
    X_train_2d = reshape_sequences_for_ml(X_train)
    X_test_2d = reshape_sequences_for_ml(X_test)

    print(f"\n[XGB] Training XGBoost (max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators})...")

    xgb_model = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        tree_method='hist',
        device='cpu',
        verbosity=1
    )

    xgb_model.fit(
        X_train_2d, y_train,
        eval_set=[(X_test_2d, y_test)],
        verbose=False
    )

    # Make predictions
    print(f"\n[XGB] Making predictions...")
    y_pred = xgb_model.predict(X_test_2d)

    # Calculate metrics
    print(f"\n{'='*60}")
    print("XGBOOST PERFORMANCE METRICS")
    print(f"{'='*60}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"✓ MSE:   {mse:.6f}")
    print(f"✓ RMSE:  {rmse:.6f}")
    print(f"✓ MAE:   {mae:.6f}")
    print(f"✓ R²:    {r2:.6f}")
    print(f"✓ MAPE:  {mape:.6f}")

    # Feature importance
    feature_importance = xgb_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:]
    print(f"\n✓ Top 10 most important features (indices): {top_features_idx}")

    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:500], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500], label='XGBoost Prediction', alpha=0.7, linewidth=2)
    plt.legend(fontsize=12)
    plt.title('XGBoost - GHI Prediction (First 500 samples)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Normalized GHI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/xgboost_predictions.png', dpi=100)
    print(f"\n✓ Plot saved: {RESULTS_DIR}/xgboost_predictions.png")

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(xgb_model, max_num_features=15)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/xgboost_feature_importance.png', dpi=100)
    print(f"\n✓ Feature importance plot saved: {RESULTS_DIR}/xgboost_feature_importance.png")

    return {
        'model': xgb_model,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }


if __name__ == "__main__":
    import os

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('/content/data/X_train.npy') # Corrected from X_test.npy
    X_test = np.load('/content/data/X_test.npy')
    y_train = np.load('/content/data/y_train.npy')
    y_test = np.load('/content/data/y_test.npy')

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create output directory
    ensure_dir(RESULTS_DIR)

    # Train models
    rf_results = train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=20)
    xgb_results = train_xgboost(X_train, X_test, y_train, y_test, max_depth=7, learning_rate=0.1, n_estimators=200)

    print(f"\n✅ RANDOM FOREST & XGBOOST TRAINING COMPLETE!")