"""
Spatio-Temporal Preprocessing Pipeline for Solar Irradiance Prediction
- Multi-site data aligned by datetime
- Spatial correlation analysis
- Spatio-temporal tensor creation
- No temporal data leakage (proper temporal split)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import pickle
from pathlib import Path
from path_utils import ensure_dir, NSRDB_DIR, SCALERS_FILE

warnings.filterwarnings('ignore')

# Site metadata
SITE_METADATA = {
    'Germany_Berlin': {'lat': 52.52, 'lon': 13.40, 'site_id': 0},
    'Egypt_Cairo': {'lat': 30.04, 'lon': 31.24, 'site_id': 1},
    'India_Delhi': {'lat': 28.61, 'lon': 77.23, 'site_id': 2},
    'India_Bangalore': {'lat': 12.97, 'lon': 77.59, 'site_id': 3},
    'India_Pune': {'lat': 18.52, 'lon': 73.86, 'site_id': 4},
    'India_Leh': {'lat': 34.15, 'lon': 77.58, 'site_id': 5}
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates (km)"""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def compute_spatial_distance_matrix():
    """Compute distance matrix between sites"""
    sites = list(SITE_METADATA.keys())
    n_sites = len(sites)
    dist_matrix = np.zeros((n_sites, n_sites))
    
    for i, site1 in enumerate(sites):
        for j, site2 in enumerate(sites):
            if i != j:
                meta1 = SITE_METADATA[site1]
                meta2 = SITE_METADATA[site2]
                dist = haversine_distance(meta1['lat'], meta1['lon'], 
                                        meta2['lat'], meta2['lon'])
                dist_matrix[i, j] = dist
    
    return dist_matrix, sites


def load_and_preprocess_single_site(file_path, site_name):
    """Load and preprocess data for a single site"""
    print(f"\n[SITE: {site_name}]")
    
    df = pd.read_csv(file_path, skiprows=2)
    print(f"  Shape: {df.shape}")
    
    # Create datetime
    df['Datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()
    
    # Select features
    relevant_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative Humidity', 
                     'Wind Speed', 'Pressure']
    existing_cols = [col for col in relevant_cols if col in df.columns]
    df_model = df[existing_cols].copy()
    
    # Handle missing values
    df_model = df_model.interpolate(method='linear', limit_direction='both')
    df_model = df_model.fillna(method='bfill').fillna(method='ffill')
    
    # Remove night values (GHI > 1)
    df_model = df_model[df_model['GHI'] > 1]
    
    # Remove outliers (IQR method)
    Q1 = df_model.quantile(0.25)
    Q3 = df_model.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((df_model < (Q1 - 1.5 * IQR)) | (df_model > (Q3 + 1.5 * IQR))).any(axis=1)
    df_model = df_model[outlier_mask]
    
    print(f"  ✓ Cleaned shape: {df_model.shape}")
    
    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_model)
    df_scaled = pd.DataFrame(scaled_data, columns=df_model.columns, index=df_model.index)
    
    # Add temporal features
    df_scaled['hour_norm'] = df_scaled.index.hour / 24.0
    df_scaled['month_norm'] = df_scaled.index.month / 12.0
    df_scaled['dayofyear_norm'] = df_scaled.index.dayofyear / 365.0
    df_scaled['dayofweek_norm'] = df_scaled.index.dayofweek / 7.0
    
    print(f"  ✓ Features: {df_scaled.shape[1]}")
    
    return df_scaled, scaler, df_model


def load_all_sites():
    """Load data from all sites and align by datetime"""
    print("\n" + "="*70)
    print("LOADING ALL SITES - SPATIO-TEMPORAL PREPROCESSING")
    print("="*70)
    
    sites_data = {}
    sites_scalers = {}
    sites_original = {}
    
    # Find and load all CSV files
    for site_name in SITE_METADATA.keys():
        site_dir = os.path.join(NSRDB_DIR, site_name)
        csv_files = sorted([f for f in os.listdir(site_dir) if f.endswith('.csv')])
        
        print(f"\n[{site_name}] Found {len(csv_files)} files: {csv_files}")
        
        all_data = []
        for csv_file in csv_files:
            csv_path = os.path.join(site_dir, csv_file)
            df, scaler, df_orig = load_and_preprocess_single_site(csv_path, csv_file)
            all_data.append(df)
            sites_scalers[site_name] = scaler
            sites_original[site_name] = df_orig
        
        # Concatenate all years for this site
        site_data = pd.concat(all_data).sort_index()
        sites_data[site_name] = site_data
        print(f"  ✓ Total records: {len(site_data)}")
    
    # Find common time period
    print("\n[ALIGNMENT] Finding common time period across sites...")
    all_indices = [set(sites_data[site].index) for site in sites_data.keys()]
    common_index = all_indices[0].intersection(*all_indices[1:])
    common_index = sorted(list(common_index))
    
    print(f"  ✓ Common period: {common_index[0]} to {common_index[-1]}")
    print(f"  ✓ Common records: {len(common_index)}")
    
    # Align all sites to common index
    aligned_data = {}
    for site_name in sites_data.keys():
        aligned_data[site_name] = sites_data[site_name].loc[common_index]
    
    return aligned_data, sites_scalers, sites_original


def compute_spatial_correlations(aligned_data):
    """Compute inter-site correlations"""
    print("\n[SPATIAL ANALYSIS] Computing site correlations...")
    
    sites = list(aligned_data.keys())
    n_sites = len(sites)
    
    # Correlation matrix for GHI
    ghi_data = np.array([aligned_data[site]['GHI'].values for site in sites])
    corr_matrix = np.corrcoef(ghi_data)
    
    # Distance matrix
    dist_matrix, _ = compute_spatial_distance_matrix()
    
    print("\n  GHI Correlation Matrix:")
    for i, site1 in enumerate(sites):
        for j, site2 in enumerate(sites):
            print(f"    {site1} <-> {site2}: {corr_matrix[i, j]:.4f} (dist: {dist_matrix[i, j]:.0f}km)")
    
    return corr_matrix, dist_matrix


def create_spatiotemporal_sequences(aligned_data, seq_len=24, horizon=1):
    """
    Create spatio-temporal sequences
    
    Args:
        aligned_data: dict of DataFrames, one per site
        seq_len: historical sequence length (hours)
        horizon: forecast horizon (hours ahead)
    
    Returns:
        X: shape (n_samples, n_sites, seq_len, n_features)
        y: shape (n_samples, n_sites) - target GHI for each site
        dates: datetime index
    """
    print(f"\n[SEQUENCES] Creating spatio-temporal sequences (seq_len={seq_len}, horizon={horizon})...")
    
    sites = sorted(list(aligned_data.keys()))
    n_sites = len(sites)
    
    # Get site data
    site_arrays = []
    datetimes = None
    
    for site in sites:
        arr = aligned_data[site].values  # (n_samples, n_features)
        if datetimes is None:
            datetimes = aligned_data[site].index
        site_arrays.append(arr)
    
    n_samples_total = site_arrays[0].shape[0]
    n_features = site_arrays[0].shape[1]
    
    # Create sequences
    X = []
    y = []
    valid_dates = []
    
    n_valid = n_samples_total - seq_len - horizon + 1
    
    for i in range(n_valid):
        # Historical sequence: [t-seq_len, ..., t-1]
        x_seq = np.zeros((n_sites, seq_len, n_features))
        for site_idx in range(n_sites):
            x_seq[site_idx] = site_arrays[site_idx][i:i+seq_len]
        
        # Target: GHI at t+horizon for all sites (first column)
        y_seq = np.array([site_arrays[site_idx][i+seq_len+horizon-1, 0] 
                         for site_idx in range(n_sites)])
        
        X.append(x_seq)
        y.append(y_seq)
        valid_dates.append(datetimes[i+seq_len+horizon-1])
    
    X = np.array(X)  # (n_samples, n_sites, seq_len, n_features)
    y = np.array(y)  # (n_samples, n_sites)
    
    print(f"  ✓ X shape: {X.shape} (samples, sites, seq_len, features)")
    print(f"  ✓ y shape: {y.shape} (samples, sites)")
    
    return X, y, np.array(valid_dates)


def temporal_train_test_split(X, y, dates, train_ratio=0.8):
    """
    Temporal split (no leakage)
    
    Args:
        X: spatio-temporal sequences
        y: targets
        dates: datetime index
        train_ratio: train/test split ratio
    
    Returns:
        Train and test data with temporal separation
    """
    print(f"\n[TEMPORAL SPLIT] Splitting data temporally (no leakage)...")
    
    n_train = int(len(X) * train_ratio)
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    dates_train, dates_test = dates[:n_train], dates[n_train:]
    
    print(f"  ✓ Train set:")
    print(f"      X: {X_train.shape}, y: {y_train.shape}")
    print(f"      Time: {dates_train[0]} to {dates_train[-1]}")
    print(f"  ✓ Test set:")
    print(f"      X: {X_test.shape}, y: {y_test.shape}")
    print(f"      Time: {dates_test[0]} to {dates_test[-1]}")
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def preprocess_spatiotemporal(seq_len=24, horizon=1, train_ratio=0.8, save=True):
    """
    Complete spatio-temporal preprocessing pipeline
    
    Args:
        seq_len: sequence length (hours)
        horizon: forecast horizon (hours)
        train_ratio: train/test split
        save: whether to save to files
    
    Returns:
        Dictionary with all data
    """
    # Load and align sites
    aligned_data, scalers, original = load_all_sites()
    
    # Spatial analysis
    corr_matrix, dist_matrix = compute_spatial_correlations(aligned_data)
    
    # Create sequences
    X, y, dates = create_spatiotemporal_sequences(aligned_data, seq_len=seq_len, horizon=horizon)
    
    # Temporal split
    X_train, X_test, y_train, y_test, dates_train, dates_test = temporal_train_test_split(
        X, y, dates, train_ratio=train_ratio
    )
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'dates_train': dates_train,
        'dates_test': dates_test,
        'scalers': scalers,
        'corr_matrix': corr_matrix,
        'dist_matrix': dist_matrix,
        'sites': sorted(list(SITE_METADATA.keys())),
        'seq_len': seq_len,
        'horizon': horizon
    }
    
    if save:
        ensure_dir('data')
        np.save('data/X_train_st.npy', X_train)
        np.save('data/X_test_st.npy', X_test)
        np.save('data/y_train_st.npy', y_train)
        np.save('data/y_test_st.npy', y_test)
        np.save('data/dates_train.npy', dates_train, allow_pickle=True)
        np.save('data/dates_test.npy', dates_test, allow_pickle=True)
        np.save('data/corr_matrix.npy', corr_matrix)
        np.save('data/dist_matrix.npy', dist_matrix)
        
        with open('data/metadata.pkl', 'wb') as f:
            pickle.dump({'sites': result['sites'], 'seq_len': seq_len, 'horizon': horizon}, f)
        
        print(f"\n{'='*70}")
        print("✓ SPATIO-TEMPORAL DATA SAVED")
        print(f"{'='*70}")
        print("  - data/X_train_st.npy")
        print("  - data/X_test_st.npy")
        print("  - data/y_train_st.npy")
        print("  - data/y_test_st.npy")
        print("  - data/corr_matrix.npy")
        print("  - data/dist_matrix.npy")
    
    return result


if __name__ == "__main__":
    dataset = preprocess_spatiotemporal(seq_len=24, horizon=1, train_ratio=0.8, save=True)
    
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETE - READY FOR MODEL TRAINING")
    print(f"{'='*70}")
    print(f"\nDataset Summary:")
    print(f"  - Training samples: {len(dataset['X_train'])}")
    print(f"  - Test samples: {len(dataset['X_test'])}")
    print(f"  - Sites: {len(dataset['sites'])}")
    print(f"  - Features per timestep: {dataset['X_train'].shape[-1]}")
    print(f"  - Sequence length: {dataset['seq_len']} hours")
    print(f"  - Forecast horizon: {dataset['horizon']} hour(s)")
