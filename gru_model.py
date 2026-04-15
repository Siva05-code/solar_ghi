"""
GRU (Gated Recurrent Unit) Model for Solar Irradiance Prediction
Baseline model for comparison
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)


def build_gru_model(input_shape, seq_len=24, num_sites=3):
    """
    Build GRU model for sequence prediction
    
    Args:
        input_shape: (n_sites, seq_len, n_features)
        seq_len: sequence length
        num_sites: number of sites
    """
    # Flatten spatial dimension into features
    model = Sequential([
        # Reshape to (batch, seq_len, n_sites * n_features)
        keras.layers.Reshape((seq_len, -1), input_shape=input_shape),
        
        # First GRU layer
        GRU(128, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second GRU layer
        GRU(64, activation='relu', return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third GRU layer
        GRU(32, activation='relu', return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(num_sites, activation='sigmoid')  # Multi-site output
    ])
    
    return model


def train_gru_model(X_train, X_test, y_train, y_test, epochs=10, batch_size=32, num_sites=3):
    """Train GRU model"""
    
    print("\n" + "="*70)
    print("GRU (GATED RECURRENT UNIT) MODEL FOR SOLAR IRRADIANCE PREDICTION")
    print("="*70)
    
    print(f"\nInput Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # Build model
    print(f"\n[Building] GRU Model...")
    model = build_gru_model(X_train.shape[1:], seq_len=X_train.shape[2], num_sites=num_sites)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    print(f"  ✓ Model built")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint('models/gru_best.h5', monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    # Train
    print(f"\n[Training] GRU Model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    print(f"\n[Saving] Model...")
    model.save('models/gru_final.h5')
    
    # Evaluate
    print(f"\n[Evaluation] GRU Model")
    print("="*70)
    
    y_pred_test = model.predict(X_test)
    
    mse = mean_squared_error(y_test.flatten(), y_pred_test.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.flatten(), y_pred_test.flatten())
    r2 = r2_score(y_test.flatten(), y_pred_test.flatten())
    mape = mean_absolute_percentage_error(y_test.flatten(), y_pred_test.flatten())
    
    print(f"\nOverall Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  MAPE: {mape:.6f}%")
    
    return model, history, {
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
    
    model, history, metrics = train_gru_model(X_train, X_test, y_train, y_test)
