import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import os
from path_utils import X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, LSTM_MODEL_FILE, RESULTS_DIR, ensure_dir
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)


def build_lstm_model(input_shape, seq_len=24, num_sites=3):
    """
    Build LSTM model for spatio-temporal sequence prediction

    Args:
        input_shape: (seq_len, n_features)
        seq_len: sequence length
        num_sites: number of geographic locations
    """
    model = Sequential([
        # Reshape input to flatten spatial dimension
        keras.layers.Reshape((seq_len, -1), input_shape=input_shape),
        
        # First LSTM layer
        LSTM(128, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        # Second LSTM layer
        LSTM(64, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),

        # Third LSTM layer
        LSTM(32, activation='relu', return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(num_sites, activation='sigmoid')  # Multi-site output
    ])

    return model


def train_lstm(X_train, X_test, y_train, y_test, epochs=50, batch_size=32, num_sites=3):
    """
    Train LSTM model for multi-site solar irradiance prediction
    """
    print("\n" + "="*60)
    print("LSTM MODEL FOR MULTI-SITE GHI PREDICTION")
    print("="*60)

    print(f"\nInput shapes:")
    print(f"  X_train: {X_train.shape} (samples, sites, seq_len, features)")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape} (samples, sites)")
    print(f"  y_test: {y_test.shape}")

    # Build model
    print(f"\n[LSTM] Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (sites, seq_len, n_features)

    model = build_lstm_model(input_shape, seq_len=X_train.shape[2], num_sites=num_sites)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )

    print(model.summary())

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Train model
    print(f"\n[LSTM] Training model (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Make predictions
    print(f"\n[LSTM] Making predictions...")
    y_pred = model.predict(X_test, verbose=0)

    # Calculate metrics
    print(f"\n{'='*60}")
    print("LSTM PERFORMANCE METRICS")
    print(f"{'='*60}")

    mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.flatten(), y_pred.flatten())
    r2 = r2_score(y_test.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_test.flatten(), y_pred.flatten())

    print(f"✓ MSE:   {mse:.6f}")
    print(f"✓ RMSE:  {rmse:.6f}")
    print(f"✓ MAE:   {mae:.6f}")
    print(f"✓ R²:    {r2:.6f}")
    print(f"✓ MAPE:  {mape:.6f}%")
    
    # Per-site metrics
    site_names = ['Germany_Berlin', 'Egypt_Cairo', 'India_Delhi']
    print(f"\nPer-Site Metrics:")
    for site_idx in range(num_sites):
        if site_idx < len(site_names):
            site_name = site_names[site_idx]
        else:
            site_name = f"Site {site_idx}"
        
        site_r2 = r2_score(y_test[:, site_idx], y_pred[:, site_idx])
        site_mae = mean_absolute_error(y_test[:, site_idx], y_pred[:, site_idx])
        print(f"  {site_name}:")
        print(f"    R²:   {site_r2:.6f}")
        print(f"    MAE:  {site_mae:.6f}")

    # Plot training history
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    plt.title('Model MAE', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(RESULTS_DIR)
    plt.savefig(f'{RESULTS_DIR}/lstm_training_history.png', dpi=100)
    print(f"\n✓ Training history plot saved: {RESULTS_DIR}/lstm_training_history.png")

    # Plot predictions
    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:500].flatten(), label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500].flatten(), label='LSTM Prediction', alpha=0.7, linewidth=2)
    plt.legend(fontsize=12)
    plt.title('LSTM - GHI Prediction (First 500 samples)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Normalized GHI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/lstm_predictions.png', dpi=100)
    print(f"\n✓ Predictions plot saved: {RESULTS_DIR}/lstm_predictions.png")

    # Save model
    model.save(LSTM_MODEL_FILE)
    print(f"\n✓ Model saved: {LSTM_MODEL_FILE}")

    return {
        'model': model,
        'history': history,
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
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(X_TRAIN_FILE)
    X_test = np.load(X_TEST_FILE)
    y_train = np.load(Y_TRAIN_FILE)
    y_test = np.load(Y_TEST_FILE)

    # Create output directories
    ensure_dir(RESULTS_DIR)
    os.makedirs('/content/models', exist_ok=True)
    
    # Get number of sites from data shape
    num_sites = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # Train LSTM
    lstm_results = train_lstm(X_train, X_test, y_train, y_test, epochs=50, batch_size=32, num_sites=num_sites)

    print(f"\n✅ LSTM TRAINING COMPLETE!")
