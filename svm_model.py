"""
Support Vector Machine (SVM) for Solar Irradiance Prediction
Baseline model for comparison
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')


def flatten_spatiotemporal_data(X):
    """Flatten spatio-temporal data for SVM
    
    Input: (n_samples, n_sites, seq_len, n_features)
    Output: (n_samples, n_sites * seq_len * n_features)
    """
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    return X_flat


def train_svm_model(X_train, X_test, y_train, y_test, kernel='rbf', C=10):
    """Train SVM model with hyperparameter options"""
    
    print("\n" + "="*70)
    print("SUPPORT VECTOR MACHINE (SVM) FOR SOLAR IRRADIANCE PREDICTION")
    print("="*70)
    
    print(f"\nInput Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # Flatten data
    print(f"\n[Processing] Flattening spatio-temporal data for SVM...")
    X_train_flat = flatten_spatiotemporal_data(X_train)
    X_test_flat = flatten_spatiotemporal_data(X_test)
    
    print(f"  ✓ X_train_flat: {X_train_flat.shape}")
    print(f"  ✓ X_test_flat:  {X_test_flat.shape}")
    
    # Standardize features
    print(f"\n[Preprocessing] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    print(f"  ✓ Features standardized")
    
    # Train SVM for each site
    print(f"\n[Training] SVM for each site (kernel='{kernel}', C={C})...")
    n_sites = y_train.shape[1]
    models = []
    predictions = []
    
    for site_idx in range(n_sites):
        print(f"  Training SVM for site {site_idx + 1}/{n_sites}...")
        
        svm = SVR(kernel=kernel, C=C, gamma='scale', epsilon=0.01)
        svm.fit(X_train_scaled, y_train[:, site_idx])
        models.append(svm)
        
        y_pred = svm.predict(X_test_scaled)
        predictions.append(y_pred)
    
    y_pred_test = np.column_stack(predictions)  # (n_samples, n_sites)
    
    # Evaluate
    print(f"\n[Evaluation] SVM Model")
    print("="*70)
    
    # Overall metrics
    mse = mean_squared_error(y_test.flatten(), y_pred_test.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.flatten(), y_pred_test.flatten())
    r2 = r2_score(y_test.flatten(), y_pred_test.flatten())
    mape = mean_absolute_percentage_error(y_test.flatten(), y_pred_test.flatten())
    
    print(f"\nOverall Metrics (averaged across all sites):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  MAPE: {mape:.6f}%")
    
    # Per-site metrics
    site_names = ['Germany_Berlin', 'Egypt_Cairo', 'India_Delhi']
    print(f"\nPer-Site Metrics:")
    for site_idx in range(n_sites):
        if site_idx < len(site_names):
            site_name = site_names[site_idx]
        else:
            site_name = f"Site {site_idx}"
        
        site_r2 = r2_score(y_test[:, site_idx], y_pred_test[:, site_idx])
        site_mae = mean_absolute_error(y_test[:, site_idx], y_pred_test[:, site_idx])
        print(f"  {site_name}:")
        print(f"    R²:   {site_r2:.6f}")
        print(f"    MAE:  {site_mae:.6f}")
    
    return models, {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred_test,
        'actual': y_test
    }


if __name__ == "__main__":
    X_train = np.load('data/X_train_st.npy')
    X_test = np.load('data/X_test_st.npy')
    y_train = np.load('data/y_train_st.npy')
    y_test = np.load('data/y_test_st.npy')
    
    models, metrics = train_svm_model(X_train, X_test, y_train, y_test, kernel='rbf', C=100)
    
    print(f"\n✅ SVM MODEL TRAINING COMPLETE")
