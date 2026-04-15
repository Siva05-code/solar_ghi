"""
Spatio-Temporal Transformer for Solar Irradiance Forecasting
- Multi-head attention for spatial dimension
- Multi-head attention for temporal dimension
- Positional encoding for both space and time
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Layer, Dense, Dropout, MultiHeadAttention, 
                                      LayerNormalization, Embedding, Reshape, Concatenate, 
                                      GlobalAveragePooling2D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)


class PositionalEncoding(Layer):
    """Positional encoding for temporal dimension"""
    def __init__(self, seq_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.pos_encoding = self.compute_pos_encoding()
    
    def compute_pos_encoding(self):
        """Compute positional encoding matrix"""
        position = np.arange(self.seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.seq_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pos_encoding


class SpatialAttention(Layer):
    """Spatial attention across multiple sites"""
    def __init__(self, embed_dim, num_heads=4):
        super(SpatialAttention, self).__init__()
        self.embed_dim = embed_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
    
    def call(self, x, training=False):
        # x: (batch, n_sites, seq_len, features)
        # Use static shapes where possible for proper weight initialization
        original_shape = tf.shape(x)
        batch_size = original_shape[0]
        n_sites = x.shape[1] if x.shape[1] is not None else original_shape[1]
        seq_len = x.shape[2] if x.shape[2] is not None else original_shape[2]
        embed_dim = x.shape[3] if x.shape[3] is not None else original_shape[3]
        
        # Reshape: (batch * seq_len, n_sites, embed_dim)
        x_reshaped = tf.reshape(x, [batch_size * seq_len, n_sites, embed_dim])
        
        # Multi-head attention
        attn = self.mha(x_reshaped, x_reshaped)
        attn = self.dropout(attn, training=training)
        attn = self.norm(x_reshaped + attn)
        
        # Reshape back: (batch, n_sites, seq_len, embed_dim)
        attn = tf.reshape(attn, [batch_size, n_sites, seq_len, embed_dim])
        return attn


class TemporalAttention(Layer):
    """Temporal attention along sequence"""
    def __init__(self, embed_dim, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
    
    def call(self, x, training=False):
        # x: (batch, n_sites, seq_len, features)
        # Use static shapes where possible for proper weight initialization
        original_shape = tf.shape(x)
        batch_size = original_shape[0]
        n_sites = x.shape[1] if x.shape[1] is not None else original_shape[1]
        seq_len = x.shape[2] if x.shape[2] is not None else original_shape[2]
        embed_dim = x.shape[3] if x.shape[3] is not None else original_shape[3]
        
        # Reshape: (batch * n_sites, seq_len, embed_dim)
        x_reshaped = tf.reshape(x, [batch_size * n_sites, seq_len, embed_dim])
        
        # Multi-head attention
        attn = self.mha(x_reshaped, x_reshaped)
        attn = self.dropout(attn, training=training)
        attn = self.norm(x_reshaped + attn)
        
        # Reshape back: (batch, n_sites, seq_len, embed_dim)
        attn = tf.reshape(attn, [batch_size, n_sites, seq_len, embed_dim])
        return attn


class FeedForwardNetwork(Layer):
    """Feed-forward network in transformer"""
    def __init__(self, embed_dim, ff_dim=256):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = Dense(ff_dim, activation='relu')
        self.dense2 = Dense(embed_dim)
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
    
    def call(self, x, training=False):
        ff = self.dense1(x)
        ff = self.dense2(ff)
        ff = self.dropout(ff, training=training)
        return self.norm(x + ff)


def build_spatiotemporal_transformer(input_shape, num_sites=3, embed_dim=64, 
                                   num_heads_spatial=4, num_heads_temporal=8,
                                   ff_dim=256, num_blocks=3, dropout=0.1):
    """
    Build Spatio-Temporal Transformer
    
    Args:
        input_shape: (seq_len, n_features)
        num_sites: number of geographic locations
        embed_dim: embedding dimension
        num_heads_spatial: number of spatial attention heads
        num_heads_temporal: number of temporal attention heads
        ff_dim: feed-forward dimension
        num_blocks: number of spatio-temporal blocks
        dropout: dropout rate
    
    Returns:
        model
    """
    seq_len, n_features = input_shape
    
    # Input: (batch, n_sites, seq_len, n_features)
    inputs = Input(shape=(num_sites, seq_len, n_features))
    
    # Project features to embedding dimension
    x = Dense(embed_dim, activation='relu')(inputs)
    
    # Stacked spatio-temporal blocks
    for _ in range(num_blocks):
        # Spatial attention
        x_spatial = SpatialAttention(embed_dim, num_heads_spatial)(x)
        
        # Temporal attention
        x_temporal = TemporalAttention(embed_dim, num_heads_temporal)(x_spatial)
        
        # Feed-forward
        x = FeedForwardNetwork(embed_dim, ff_dim)(x_temporal)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)  # (batch, sites, features)
    x = Flatten()(x)  # (batch, sites * features)
    
    # Dense layers for site-specific predictions
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    
    # Output: predictions for all sites
    outputs = Dense(num_sites, activation='sigmoid')(x)  # (batch, num_sites)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_spatiotemporal_transformer(X_train, X_test, y_train, y_test, 
                                    epochs=10, batch_size=32, num_sites=3):
    """Train spatio-temporal transformer for GHI prediction"""
    
    print("\n" + "="*70)
    print("SPATIO-TEMPORAL TRANSFORMER FOR SOLAR IRRADIANCE FORECASTING")
    print("="*70)
    
    print(f"\nInput Shapes:")
    print(f"  X_train: {X_train.shape} (samples, sites, seq_len, features)")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape} (samples, sites)")
    print(f"  y_test:  {y_test.shape}")
    
    # Build model
    print(f"\n[Building] Spatio-Temporal Transformer...")
    input_shape = (X_train.shape[2], X_train.shape[3])  # (seq_len, n_features)
    
    model = build_spatiotemporal_transformer(
        input_shape,
        num_sites=num_sites,
        embed_dim=64,
        num_heads_spatial=4,
        num_heads_temporal=8,
        ff_dim=256,
        num_blocks=3,
        dropout=0.1
    )
    
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
        ModelCheckpoint('models/transformer_st_best.h5', monitor='val_loss', 
                       save_best_only=True, verbose=0)
    ]
    
    # Train
    print(f"\n[Training] Spatio-Temporal Transformer...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    print(f"\n[Saving] Model...")
    model.save('models/transformer_st_final.h5')
    
    # Evaluate
    print(f"\n[Evaluation] Spatio-Temporal Transformer")
    print("="*70)
    
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics (averaged across sites)
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
    for site_idx, site_name in enumerate(site_names):
        site_r2 = r2_score(y_test[:, site_idx], y_pred_test[:, site_idx])
        site_mae = mean_absolute_error(y_test[:, site_idx], y_pred_test[:, site_idx])
        print(f"  {site_name}:")
        print(f"    R²:   {site_r2:.6f}")
        print(f"    MAE:  {site_mae:.6f}")
    
    # Plot
    print(f"\n[Plotting] Training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Spatio-Temporal Transformer - Training History')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Spatio-Temporal Transformer - MAE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/transformer_st_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
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
    # Load data
    X_train = np.load('data/X_train_st.npy')
    X_test = np.load('data/X_test_st.npy')
    y_train = np.load('data/y_train_st.npy')
    y_test = np.load('data/y_test_st.npy')
    
    # Train model
    model, history, metrics = train_spatiotemporal_transformer(
        X_train, X_test, y_train, y_test,
        epochs=10, batch_size=32, num_sites=3
    )
    
    print(f"\n✅ SPATIO-TEMPORAL TRANSFORMER TRAINING COMPLETE")
