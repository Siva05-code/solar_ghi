import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
from path_utils import X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE, TRANSFORMER_BEST_FILE, TRANSFORMER_MODEL_FILE, RESULTS_DIR, ensure_dir
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)


class TransformerBlock(Layer):
    """
    Transformer encoder block with multi-head attention
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(input_shape, num_transformer_blocks=4, head_size=256, num_heads=8, ff_dim=128, dropout=0.1):
    """
    Build Transformer model for sequence prediction

    Args:
        input_shape: (seq_len, n_features)
        num_transformer_blocks: number of transformer encoder blocks
        head_size: size of attention heads
        num_heads: number of attention heads
        ff_dim: dimension of feed-forward network
        dropout: dropout rate
    """
    inputs = Input(shape=input_shape)

    # Project input features to the embedding dimension (head_size)
    x = Dense(head_size, activation="relu")(inputs)

    # Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(head_size, num_heads, ff_dim, dropout)(x)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers for prediction
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)  # GHI normalized to [0,1]

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_transformer(X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
    """
    Train Transformer model for GHI prediction
    """
    print("\n" + "="*60)
    print("TRANSFORMER MODEL FOR GHI PREDICTION (SPATIAL-TEMPORAL)")
    print("="*60)

    print(f"\nInput shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    # Build model
    print(f"\n[Transformer] Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, n_features)

    model = build_transformer_model(
        input_shape,
        num_transformer_blocks=4,
        head_size=64,
        num_heads=8,
        ff_dim=128,
        dropout=0.1
    )

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
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        '/content/models/transformer_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train model
    print(f"\n[Transformer] Training model (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # Make predictions
    print(f"\n[Transformer] Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()

    # Calculate metrics
    print(f"\n{'='*60}")
    print("TRANSFORMER PERFORMANCE METRICS")
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
    plt.title('Transformer - Model Loss', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    plt.title('Transformer - Model MAE', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/results/transformer_training_history.png', dpi=100)
    print(f"\n✓ Training history plot saved: /content/results/transformer_training_history.png")

    # Plot predictions
    plt.figure(figsize=(14, 5))
    plt.plot(y_test[:500], label='Actual', alpha=0.7, linewidth=2, marker='o', markersize=2)
    plt.plot(y_pred[:500], label='Transformer Prediction', alpha=0.7, linewidth=2, marker='s', markersize=2)
    plt.legend(fontsize=12)
    plt.title('Transformer - GHI Prediction (First 500 samples)', fontsize=14)
    plt.xlabel('Sample')
    plt.ylabel('Normalized GHI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/content/results/transformer_predictions.png', dpi=100)
    print(f"\n✓ Predictions plot saved: /content/results/transformer_predictions.png")

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred[:1000], residuals[:1000], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residual Plot', fontsize=12)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Residuals Distribution', fontsize=12)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/results/transformer_residuals.png', dpi=100)
    print(f"\n✓ Residuals plot saved: /content/results/transformer_residuals.png")

    # Save model
    model.save('/content/models/transformer_model.h5')
    print(f"\n✓ Model saved: /content/models/transformer_model.h5")

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
    import os

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('/content/data/X_train.npy')
    X_test = np.load('/content/data/X_test.npy')
    y_train = np.load('/content/data/y_train.npy')
    y_test = np.load('/content/data/y_test.npy')

    # Create output directories
    os.makedirs('/content/results', exist_ok=True)
    os.makedirs('/content/models', exist_ok=True)

    # Train Transformer
    transformer_results = train_transformer(X_train, X_test, y_train, y_test, epochs=100, batch_size=32)

    print(f"\n✅ TRANSFORMER TRAINING COMPLETE!")