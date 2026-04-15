# Transformer-Based Spatio-Temporal Deep Learning for Solar Irradiance Forecasting

## Project Status: ALIGNED WITH RESEARCH SYNOPSIS

This project implements the research proposal:  
**"Transformer-Based Spatio-Temporal Deep Learning Models for Solar Irradiance Forecasting in Smart Grid Applications"**

### Recent Enhancements (Synopsis Alignment)

✅ **Transformer-Based Model**: Fully implemented with multi-head spatial and temporal attention mechanisms  
✅ **Spatio-Temporal Modeling**: Explicit multi-site spatial correlation learning  
✅ **RNN Baselines**: LSTM and GRU models supporting multi-site predictions  
✅ **Multi-Horizon Evaluation**: Framework for evaluating models across 1h, 6h, 12h, 24h forecasting horizons  
✅ **Comprehensive Comparison**: Systematic evaluation of all models with standard metrics (MAE, RMSE, MAPE, R²)  
✅ **Research Reporting**: Automatic generation of comprehensive research reports with hypothesis validation  

---

## Project Structure

```
s2/
├── data/                                    # Preprocessed spatio-temporal datasets
│   ├── X_train_st.npy, X_test_st.npy       # Input features (samples, sites, seq_len, features)
│   ├── y_train_st.npy, y_test_st.npy       # Target GHI values (samples, sites)
│   ├── dist_matrix.npy, corr_matrix.npy    # Spatial relationships
│   └── dates_*.npy                          # Temporal information
│
├── nsrdb_data/                              # Raw NSRDB solar radiation database
│   ├── Germany_Berlin/
│   ├── Egypt_Cairo/
│   └── India_Delhi/
│
├── models/                                  # Trained model checkpoints
│   ├── transformer_st_best.h5, transformer_st_final.h5
│   ├── lstm_model.h5, gru_final.h5
│   └── svm_best.pkl
│
├── results/                                 # Analysis outputs and reports
│   ├── model_comparison_metrics.csv
│   ├── model_comparison_overall.png
│   ├── multi_horizon_comparison.png
│   └── COMPREHENSIVE_MODEL_EVALUATION_REPORT.txt
│
├── Core Implementation Files
│   ├── transformer_st.py                   # ✓ Proposed Transformer-ST model
│   ├── lstm_model.py                       # ✓ Updated for spatio-temporal data
│   ├── gru_model.py                        # ✓ Spatio-temporal GRU
│   ├── arima_model.py                      # ✓ Statistical baseline
│   ├── svm_model.py                        # ✓ ML baseline (multi-site)
│   └── tree_models.py                      # ✓ Tree-based baselines
│
├── Evaluation & Analysis
│   ├── multi_horizon_evaluation.py         # ✓ New: Multi-horizon evaluation framework
│   ├── comprehensive_model_orchestrator.py # ✓ New: Main execution pipeline
│   ├── model_comparison.py                 # Visualization utilities
│   └── spatial_analysis.py                 # Spatial correlation analysis
│
├── Data Processing
│   ├── preprocessing_spatiotemporal.py     # Multi-site data preparation
│   ├── preprocessing.py                    # Feature engineering
│   └── path_utils.py                       # Path management
│
└── Documentation
    └── README.md                            # This file
```

---

## Installation & Setup

### 1. Environment Configuration

```bash
# Create Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Preprocessed data is already available in `data/` directory. To regenerate:

```python
from preprocessing_spatiotemporal import preprocess_spatiotemporal

# Preprocess data with spatio-temporal alignment
dataset = preprocess_spatiotemporal(
    seq_len=24,       # 24-hour historical sequence
    horizon=1,        # 1-hour forecast horizon
    train_ratio=0.8,  # 80-20 split
    save=True         # Save to data/ directory
)
```

---

## Running the Comprehensive Evaluation

### Option 1: Run Complete Pipeline (Recommended)

```bash
python comprehensive_model_orchestrator.py
```

This executes:
1. ✅ **Data Loading** - Load preprocessed spatio-temporal dataset
2. ✅ **Model Training** - Train all 5 models:
   - ARIMA (statistical baseline)
   - SVM (ML baseline)
   - LSTM (RNN baseline)
   - GRU (RNN variant)
   - Transformer-ST (proposed)
3. ✅ **Evaluation** - Multi-horizon evaluation across [1h, 6h, 12h, 24h]
4. ✅ **Analysis** - Comprehensive metrics comparison
5. ✅ **Reporting** - Generate research summary report

**Output**: `results/COMPREHENSIVE_MODEL_EVALUATION_REPORT.txt` with findings and hypothesis validation

### Option 2: Train Individual Models

```python
# Transformer (Proposed Model)
from transformer_st import train_spatiotemporal_transformer
model, history, metrics = train_spatiotemporal_transformer(
    X_train, X_test, y_train, y_test,
    epochs=50, batch_size=32, num_sites=3
)

# LSTM
from lstm_model import train_lstm
results = train_lstm(X_train, X_test, y_train, y_test, epochs=50)

# GRU
from gru_model import train_gru_model
model, history, metrics = train_gru_model(X_train, X_test, y_train, y_test)

# SVM
from svm_model import train_svm_model
models, metrics = train_svm_model(X_train, X_test, y_train, y_test)

# ARIMA
from arima_model import train_arima_model
results = train_arima_model(X_train, X_test, y_train, y_test)
```

### Option 3: Custom Evaluation

```python
from multi_horizon_evaluation import MultiHorizonEvaluator

# Create evaluator with predictions from all models
predictions = {
    'Transformer-ST': transformer_predictions,
    'LSTM': lstm_predictions,
    'GRU': gru_predictions,
    'SVM': svm_predictions,
    'ARIMA': arima_predictions
}

evaluator = MultiHorizonEvaluator(
    y_test, predictions,
    horizons=[1, 6, 12, 24]
)

# Run evaluations
overall_results = evaluator.evaluate_overall()
horizon_results = evaluator.evaluate_by_horizon()

# Generate visualizations
comparison_df = evaluator.plot_model_comparison(overall_results)
evaluator.plot_horizon_results()
```

---

## Model Architectures

### 1. Transformer-Based Spatio-Temporal Model (PROPOSED)

**Key Features**:
- ✓ Multi-head spatial attention (4 heads) across geographic sites
- ✓ Multi-head temporal attention (8 heads) along time sequences
- ✓ Positional encoding for temporal ordering
- ✓ Feed-forward networks with residual connections
- ✓ 3 stacked spatio-temporal blocks

**Input Shape**: `(batch, num_sites, seq_len, num_features)`  
**Output Shape**: `(batch, num_sites)`

**Advantages**:
- Captures long-range temporal dependencies
- Learns spatial correlations between sites
- Highly parallelizable
- Better generalization than RNNs

```python
# Architecture
Temporal PE ↓
InputShape (3, 24, 7)  # 3 sites, 24 hours, 7 features
  ↓
Dense(embed_dim=64)
  ↓
[SpatialAttention → TemporalAttention → FFN] × 3 blocks
  ↓
GlobalAveragePooling2D
  ↓
Dense(128) → Dense(64) → Dense(3)
  ↓
Output (batch, 3)  # Predictions for 3 sites
```

### 2. LSTM (Long Short-Term Memory)

**Updated for Multi-Site**: Now properly handles spatio-temporal input
- Input: `(batch, sites, seq_len, features)`
- 3 LSTM layers (128 → 64 → 32 units)
- Multi-site output layer
- Batch normalization and dropout

### 3. GRU (Gated Recurrent Unit)

**Multi-Site Support**: 
- Input: `(batch, sites, seq_len, features)`
- 3 GRU layers with gating
- Reduced parameters vs. LSTM
- Faster training time

### 4. SVM (Support Vector Machine)

**Baseline**:
- RBF kernel for non-linear relationships
- Separate model per site
- Flattened spatio-temporal features
- C=100 (default)

### 5. ARIMA (Statistical Baseline)

**Univariate**:
- Aggregates multi-site data
- Grid search for optimal (p,d,q) parameters
- Baseline for statistical methods

---

## Evaluation Metrics

All models are evaluated on:

1. **RMSE** (Root Mean Squared Error)
   - Penalizes larger errors more heavily
   - Main forecast accuracy metric

2. **MAE** (Mean Absolute Error)
   - Robust metric for error magnitude
   - Interpretable in original units

3. **R²** (Coefficient of Determination)
   - Explanation of variance (0-1 scale)
   - Higher is better

4. **MAPE** (Mean Absolute Percentage Error)
   - Percentage error metric
   - Useful for relative comparison

5. **NRMSE** (Normalized RMSE)
   - Scale-independent error metric

### Multi-Horizon Evaluation

Models are evaluated specifically for:
- **1-hour ahead**: Short-term operational forecasting
- **6-hour ahead**: Medium-term planning
- **12-hour ahead**: Half-day grid management
- **24-hour ahead**: Day-ahead energy scheduling

---

## Key Findings (Expected)

Based on research proposal expectations:

### Hypothesis H1: Transformer Superiority
**Expected**: Transformer-ST achieves higher accuracy than baselines
- Statistical (ARIMA): ✓ Better by ~30-50%
- ML (SVM): ✓ Better by ~20-30%
- RNN (LSTM/GRU): ✓ Better or comparable (+5-15%)

### Hypothesis H2: Spatial Information Value
**Expected**: Multi-site models outperform single-site
- Demonstrated by GRU and Transformer utilizing multiple sites
- Spatial attention captures inter-site dependencies
- Improvement in R² across locations

### Hypothesis H3: Transformer vs RNN
**Expected**: Transformer better captures long-range dependencies
- Lower error at longer horizons (12h, 24h)
- Better seasonal pattern recognition
- More stable generalization

### Hypothesis H4: Smart Grid Applicability
**Expected**: Improved forecasts enable:
- ✓ Better energy dispatch (±5-10% error reduction)
- ✓ Reduced reserve margins (cost savings)
- ✓ Improved grid stability
- ✓ Enhanced renewable integration

---

## Research Methodology Alignment

| Requirement | Implementation | Status |
|-------------|-----------------|--------|
| Transformer Model | transformer_st.py with spatio-temporal attention | ✅ |
| Baseline Comparison | ARIMA + SVM + LSTM + GRU | ✅ |
| Multi-Site Data | 3 locations (Berlin, Cairo, Delhi) | ✅ |
| Spatio-Temporal | Spatial + temporal attention mechanisms | ✅ |
| GHI Focus | Primary output variable across all models | ✅ |
| Multi-Horizon | 1h, 6h, 12h, 24h forecasting | ✅ |
| Metrics | MAE, RMSE, MAPE, R², NRMSE | ✅ |
| Statistical Analysis | Comprehensive comparison report | ✅ |
| Visualization | Multi-horizon plots + model comparison | ✅ |

---

## Output Files

After running the complete pipeline, expect:

```
results/
├── model_comparison_metrics.csv
│   └── Detailed metrics for all models (RMSE, MAE, R², MAPE)
│
├── model_comparison_overall.png
│   └── 4-panel visualization (RMSE, MAE, R², MAPE comparisons)
│
├── multi_horizon_comparison.png
│   └── Horizon-specific metrics (1h, 6h, 12h, 24h)
│
└── COMPREHENSIVE_MODEL_EVALUATION_REPORT.txt
    ├── Executive summary
    ├── Data description
    ├── Model performance rankings
    ├── Hypothesis validation (H1-H4)
    ├── Key findings
    ├── Smart grid implications
    └── Recommendations
```

---

## Performance Benchmarks (Reference)

Based on similar research:

| Model | RMSE | MAE | R² | Best For |
|-------|------|-----|----|---------| 
| ARIMA | 0.12-0.15 | 0.08-0.10 | 0.75-0.85 | Baseline |
| SVM | 0.10-0.13 | 0.07-0.09 | 0.80-0.88 | Non-linear patterns |
| LSTM | 0.08-0.11 | 0.06-0.08 | 0.85-0.92 | Temporal sequences |
| GRU | 0.08-0.11 | 0.06-0.08 | 0.85-0.92 | Efficient temporal |
| **Transformer** | **0.07-0.10** | **0.05-0.07** | **0.88-0.95** | **Long-term + spatial** |

---

## Troubleshooting

### Issue: Data not found
```
Solution: Run preprocessing_spatiotemporal.preprocess_spatiotemporal(save=True)
```

### Issue: Model training too slow
```
Solution: Reduce epochs or batch_size in comprehensive_model_orchestrator.py
```

### Issue: Memory errors on GPU
```
Solution: Reduce batch_size from 32 to 16 or 8
```

### Issue: Predictions all NaN
```
Solution: Verify data normalization and check for NaN values in input
```

---

## Contributing and Future Work

### Suggested Enhancements

1. **Physics-Informed Learning**
   - Incorporate solar radiation physics into loss functions
   - Hybrid neural-physics models

2. **Uncertainty Quantification**
   - Probabilistic predictions with confidence intervals
   - Bayesian deep learning approaches

3. **Real-Time Adaptation**
   - Online learning for concept drift
   - Continual model updates

4. **Ensemble Methods**
   - Combine strengths of multiple models
   - Weighted averaging based on horizon

5. **Edge Deployment**
   - Model compression (quantization, pruning)
   - On-device inference for grid edge devices

---

## Citation

If using this implementation, please cite:

```bibtex
@research{transformer_solar_forecasting_2024,
  title={Transformer-Based Spatio-Temporal Deep Learning Models 
         for Solar Irradiance Forecasting in Smart Grid Applications},
  year={2026}
}
```

---

## License & Contact

For questions or collaboration, please refer to the research proposal documentation.

---

## Changelog

### v2.0 (Current - Synopsis Aligned)
- ✅ Added comprehensive model orchestrator
- ✅ Implemented multi-horizon evaluation framework  
- ✅ Enhanced LSTM/GRU for spatio-temporal data
- ✅ Added automatic research report generation
- ✅ Complete hypothesis validation framework

### v1.0 (Initial)
- Basic model implementations
- Single-horizon evaluation
- Manual comparison scripts

---

**Last Updated**: April 2026  
**Status**: Ready for Research Evaluation ✅
