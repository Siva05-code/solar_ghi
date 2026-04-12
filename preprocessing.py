"""
Preprocessing pipeline for solar irradiance prediction
Target: GHI (Global Horizontal Irradiance)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import pickle
from path_utils import ensure_dir, NSRDB_DIR, X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, SCALERS_FILE
warnings.filterwarnings('ignore')

def load_and_preprocess_csv(file_path):
    """
    Load CSV with correct header structure and preprocess
    
    Row 1: Metadata column names
    Row 2: Site metadata
    Row 3: Actual data header
    Row 4 onward: Real observations
    """
    print(f"\n{'='*60}")
    print(f"Loading: {file_path}")
    print(f"{'='*60}")
    
    # Load with correct header (skiprows=2 means skip first 2 rows)
    df = pd.read_csv(file_path, skiprows=2)
    print(f"✓ Shape after loading: {df.shape}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Step 1: Create datetime column
    print("\n[Step 1] Creating datetime column...")
    df['Datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute']])
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()
    print(f"✓ Datetime range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Select relevant columns for GHI prediction
    print("\n[Step 2] Selecting relevant features...")
    relevant_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative Humidity', 
                     'Wind Speed', 'Pressure']
    
    # Check which columns exist
    existing_cols = [col for col in relevant_cols if col in df.columns]
    print(f"✓ Available columns: {existing_cols}")
    
    df_model = df[existing_cols].copy()
    initial_shape = df_model.shape[0]
    print(f"✓ Selected {len(existing_cols)} features, {initial_shape} rows")
    
    # Step 3: Handle missing values
    print("\n[Step 3] Handling missing values...")
    missing_before = df_model.isnull().sum().sum()
    print(f"✓ Missing values before: {missing_before}")
    
    # Use interpolation first, then forward fill
    df_model = df_model.interpolate(method='linear', limit_direction='both')
    df_model = df_model.fillna(method='bfill').fillna(method='ffill')
    
    missing_after = df_model.isnull().sum().sum()
    print(f"✓ Missing values after: {missing_after}")
    
    # Step 4: Remove night values (GHI = 0)
    print("\n[Step 4] Removing night values...")
    df_model = df_model[df_model['GHI'] > 1]  # Small threshold to avoid noise
    rows_after_night_removal = df_model.shape[0]
    print(f"✓ Rows after removing night: {rows_after_night_removal} (removed {initial_shape - rows_after_night_removal})")
    
    # Step 5: Remove outliers using IQR method
    print("\n[Step 5] Removing outliers...")
    rows_before_outlier = df_model.shape[0]
    
    Q1 = df_model.quantile(0.25)
    Q3 = df_model.quantile(0.75)
    IQR = Q3 - Q1
    
    # Remove rows where any value is outside 1.5*IQR
    outlier_mask = ~((df_model < (Q1 - 1.5 * IQR)) | (df_model > (Q3 + 1.5 * IQR))).any(axis=1)
    df_model = df_model[outlier_mask]
    
    rows_removed = rows_before_outlier - df_model.shape[0]
    print(f"✓ Rows removed: {rows_removed}")
    print(f"✓ Final shape: {df_model.shape}")
    
    # Step 6: Normalize data using MinMaxScaler
    print("\n[Step 6] Normalizing data (MinMaxScaler)...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_model)
    df_scaled = pd.DataFrame(scaled_data, columns=df_model.columns, index=df_model.index)
    print(f"✓ Data normalized to [0, 1]")
    print(f"✓ GHI range: [{df_scaled['GHI'].min():.4f}, {df_scaled['GHI'].max():.4f}]")
    
    # Step 7: Add temporal features
    print("\n[Step 7] Adding temporal features...")
    df_scaled['hour'] = df_scaled.index.hour
    df_scaled['month'] = df_scaled.index.month
    df_scaled['dayofyear'] = df_scaled.index.dayofyear
    df_scaled['dayofweek'] = df_scaled.index.dayofweek
    
    # Normalize temporal features
    df_scaled['hour'] = df_scaled['hour'] / 24.0
    df_scaled['month'] = df_scaled['month'] / 12.0
    df_scaled['dayofyear'] = df_scaled['dayofyear'] / 365.0
    df_scaled['dayofweek'] = df_scaled['dayofweek'] / 7.0
    
    print(f"✓ Added 4 temporal features")
    print(f"✓ Final shape: {df_scaled.shape}")
    print(f"✓ Final columns: {df_scaled.columns.tolist()}")
    
    return df_scaled, scaler, df_model


def create_sequences(data, seq_len=24):
    """
    Create sequences for training transformer/LSTM models
    
    Input: Past seq_len hours → Output: Next hour GHI
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        seq_len: sequence length (default 24 hours)
    
    Returns:
        X: sequence features (n_sequences, seq_len, n_features)
        y: target values (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  # GHI is first column (target)
    
    return np.array(X), np.array(y)


def preprocess_dataset(csv_files, seq_len=24, train_split=0.8):
    """
    Process multiple CSV files and prepare train/test data
    
    Args:
        csv_files: list of CSV file paths
        seq_len: sequence length for transformer/LSTM
        train_split: train/test split ratio
    
    Returns:
        Dictionary with training and test data
    """
    all_sequences = []
    all_targets = []
    all_scalers = {}
    
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Process each file
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df_scaled, scaler, df_original = load_and_preprocess_csv(csv_file)
            all_scalers[csv_file] = (scaler, df_original)
            
            # Create sequences
            X, y = create_sequences(df_scaled.values, seq_len=seq_len)
            all_sequences.append(X)
            all_targets.append(y)
            
            print(f"✓ Sequences created: X shape = {X.shape}, y shape = {y.shape}")
    
    # Combine all data
    X_combined = np.concatenate(all_sequences, axis=0)
    y_combined = np.concatenate(all_targets, axis=0)
    
    print(f"\n{'='*60}")
    print("COMBINED DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"✓ Total sequences: {X_combined.shape[0]}")
    print(f"✓ Sequence length: {X_combined.shape[1]} timesteps")
    print(f"✓ Features: {X_combined.shape[2]}")
    print(f"✓ Target shape: {y_combined.shape}")
    
    # Train-test split
    split_idx = int(len(X_combined) * train_split)
    
    X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
    y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
    
    print(f"\n{'='*60}")
    print("TRAIN-TEST SPLIT (80-20)")
    print(f"{'='*60}")
    print(f"✓ Train set: X {X_train.shape}, y {y_train.shape}")
    print(f"✓ Test set:  X {X_test.shape}, y {y_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scalers': all_scalers,
        'seq_len': seq_len
    }


if __name__ == "__main__":
    # Find all CSV files
    csv_files = []
    
    for root, dirs, files in os.walk(NSRDB_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    csv_files = sorted(csv_files)
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Preprocess all data
    dataset = preprocess_dataset(csv_files, seq_len=24, train_split=0.8)
    
    # Save preprocessed data
    print(f"\n{'='*60}")
    print("SAVING PREPROCESSED DATA")
    print(f"{'='*60}")
    ensure_dir('data')
    np.save(X_TRAIN_FILE, dataset['X_train'])
    np.save(X_TEST_FILE, dataset['X_test'])
    np.save(Y_TRAIN_FILE, dataset['y_train'])
    np.save(Y_TEST_FILE, dataset['y_test'])
    
    # Save scalers for inverse transformation
    with open(SCALERS_FILE, 'wb') as f:
        pickle.dump(dataset['scalers'], f)
    
    print(f"✓ Saved to {os.path.dirname(X_TRAIN_FILE)}/")
    print(f"\n✅ PREPROCESSING COMPLETE!")
