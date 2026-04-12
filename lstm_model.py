import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
from path_utils import X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, LSTM_MODEL_FILE, RESULTS_DIR, ensure_dir
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)


def build_lstm_model(input_shape, seq_len=24):
    """
    Build LSTM model for sequence prediction

    Args:
        input_shape: (seq_len, n_features)
        seq_len: sequence length
    """
    model = Sequential([
        # First LSTM layer
        LSTM(128, activation='relu', return_sequences=True, input_shape=input_shape),
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
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # GHI is normalized to [0,1]
    ])

    return model


def train_lstm(X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    """
    Train LSTM model
    """
    print("\n" + "="*60)
    print("LSTM MODEL FOR GHI PREDICTION")
    print("="*60)

    print(f"\nInput shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    # Build model
    print(f"\n[LSTM] Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_features)

    model = build_lstm_model(input_shape)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
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
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Make predictions
    print(f"\n[LSTM] Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()

    # Calculate metrics
    print(f"\n{'='*60}")
    print("LSTM PERFORMANCE METRICS")
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
    plt.plot(y_test[:500], label='Actual', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500], label='LSTM Prediction', alpha=0.7, linewidth=2)
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

    # Train LSTM
    lstm_results = train_lstm(X_train, X_test, y_train, y_test, epochs=50, batch_size=32)

    print(f"\n✅ LSTM TRAINING COMPLETE!")
