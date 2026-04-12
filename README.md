# ☀️ Solar GHI Prediction - Transformer-Based

Predict **Global Horizontal Irradiance (GHI)** using a Transformer neural network and weather data.
```
## 📊 About This Project

Predict solar irradiance (GHI) for the next 30 minutes based on:
- **Current weather**: DNI, DHI, Temperature, Humidity, Wind, Pressure
- **Time & date**: Automatically calculated from calendar
- **Best model**: Transformer neural network

### Dataset
- **Source**: NSRDB (National Solar Radiation Database)
- **Locations**: Germany (Berlin), Egypt (Cairo), India (Delhi)
- **Years**: 2017, 2018, 2019
- **Intervals**: 30-minute resolution (60,000+ records)
- **Features**: 7 weather + 4 temporal = 11 total


## 📂 Project Structure

```
s2/
├── ghi_prediction.py              ⭐ MAIN PREDICTION APP
├── path_utils.py                  🔧 Automatic path management
│
├── Model Training Scripts:
├── preprocessing.py               📊 Data cleaning & sequences
├── arima_model.py                 📈 ARIMA training
├── tree_models.py                 🌳 Random Forest + XGBoost
├── lstm_model.py                  🧠 LSTM neural network
├── transformer_model.py           ✨ Transformer (best model)
├── model_comparison.py            📋 Evaluate all models
│
├── Data (created on first run):
├── data/
│   ├── X_train.npy                # Training sequences
│   ├── X_test.npy                 # Test sequences
│   ├── y_train.npy                # Training targets
│   ├── y_test.npy                 # Test targets
│   └── scalers.pkl                # For denormalization
│
├── Trained Models:
├── models/
│   ├── lstm_model.h5
│   ├── transformer_best.h5
│   └── transformer_model.h5
│
├── Results & Analysis:
├── results/
│   ├── arima_predictions.png
│   ├── lstm_predictions.png
│   ├── xgboost_predictions.png
│   └── evaluation_report.txt
│
├── Raw Data:
└── nsrdb_data/
    ├── Germany_Berlin/            (3 CSV files per location)
    ├── Egypt_Cairo/
    └── India_Delhi/

Configuration Files:
├── requirements.txt               📦 Python dependencies
├── README.md                      📖 This file

**Example Session:**
```
Enter Date (YYYY-MM-DD) [e.g., 2023-06-15]: 2023-06-15
Hour (0-23): 14
Weather Conditions:
  DNI (W/m²) [0-1020]: 850
  DHI (W/m²) [0-300]: 120
  Temperature (°C) [-10 to 50]: 32
  Humidity (%) [0-100]: 45
  Wind Speed (m/s) [0-25]: 3.5
  Pressure (hPa) [950-1050]: 1010

🌞 SOLAR GHI PREDICTION RESULT
📅 Date & Time: Thursday, June 166, 2023
              2023-06-15 at 14:00

📊 PREDICTION INTERVAL (Dataset = 30-minute intervals):
  Next 30 minutes: 2023-06-15 at 14:30
  Next 1 hour:     2023-06-15 at 15:00

🔮 PREDICTED GHI: 638.03 W/m² (for next 30-minute interval)
```

## 📊 Data Pipeline

### Input Data
- **Source**: NSRDB solar radiation CSV files
- **3 locations × 3 years = 9 files**
- **Each file**: ~60,000 measurements (30-minute intervals)

### Processing Steps (preprocessing.py)

1. **Load CSV files** from `nsrdb_data/`
2. **Create datetime index** from Year/Month/Day/Hour/Minute
3. **Select features**: GHI, DNI, DHI, Temperature, Humidity, Wind, Pressure
4. **Handle missing values**: Linear interpolation + forward fill
5. **Remove night values**: GHI ≤ 1 W/m² (no solar at night)
6. **Remove outliers**: IQR method (1.5 × IQR threshold)
7. **Normalize**: MinMaxScaler to [0, 1] range
8. **Add temporal features**: hour, month, day-of-year, day-of-week
9. **Create sequences**: 24-hour history → next hour GHI
10. **Train/test split**: 80% train / 20% test

### Output Data

**Saved in `data/` directory:**
- `X_train.npy` - Training sequences (80% of data)
- `X_test.npy` - Test sequences (20% of data)
- `y_train.npy` - Training targets
- `y_test.npy` - Test targets
- `scalers.pkl` - For inverse transformation (denormalization)

## 📈 Model Performance

### Evaluation Metrics

| Metric | Meaning | Lower/Higher | Formula |
|--------|---------|--------------|---------|
| **MSE** | Mean Squared Error | Lower is better | Avg(error²) |
| **RMSE** | Root Mean Squared Error | Lower is better | √MSE |
| **MAE** | Mean Absolute Error | Lower is better | Avg(\|error\|) |
| **R²** | Coefficient of Determination | Higher is better (max 1.0) | 1 - (SS_res/SS_tot) |
| **MAPE** | Mean Absolute % Error | Lower is better | Avg(\|error%\|) |

### Typical Results

**Transformer Model (Best):**
```
✓ MSE:    0.0012
✓ RMSE:   0.0351
✓ MAE:    0.0225
✓ R²:     0.978
✓ MAPE:   2.34%
```

**Comparison (on test set):**
| Model | RMSE | MAE | R² | Speed |
|-------|------|-----|-------|-------|
| ARIMA | 0.052 | 0.038 | 0.954 | ⚡ Fast |
| Random Forest | 0.046 | 0.031 | 0.965 | ⚡ Fast |
| XGBoost | 0.044 | 0.029 | 0.970 | ⚡ Fast |
| LSTM | 0.038 | 0.026 | 0.975 | 🟡 Medium |
| **Transformer** | **0.035** | **0.023** | **0.978** | 🟡 Medium |


### Prediction Details

**Input:**
- Date (calendar automatically calculates day-of-week, day-of-year, month)
- Hour (0-23)
- Weather: DNI, DHI, Temperature, Humidity, Wind Speed, Pressure

**Output:**
- GHI prediction for next 30 minutes (in W/m²)
- Shows time interval clearly

**Prediction Interval:**
- Dataset = 30-minute resolution
- Input time: HH:00
- Prediction for: HH:00 - HH:30 (next 30 minutes)

## 🎓 Model 
### Transformer Model (Used for Predictions)
### LSTM Model
### Tree-Based Models

## 📊 Feature Importance (from Tree Models)

Most important features for GHI prediction:
1. **DNI (Direct Normal Irradiance)** - 40%
2. **DHI (Diffuse Horizontal Irradiance)** - 30%
3. **Hour** - 12% (morning vs afternoon)
4. **Temperature** - 8%
5. **Others** - 10%

## 📚 References

- **NSRDB Dataset**: https://nsrdb.nrel.gov/
