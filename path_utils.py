"""
Path utilities for robust file handling
Converts absolute paths to relative paths for GitHub compatibility
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


def get_path(relative_path):
    """
    Get absolute path from relative path
    
    Args:
        relative_path: Path relative to project root (e.g., 'data/X_train.npy')
    
    Returns:
        Absolute path as string
    """
    return str(PROJECT_ROOT / relative_path)


def ensure_dir(directory_path):
    """
    Ensure directory exists, create if needed
    
    Args:
        directory_path: Relative or absolute directory path
    
    Returns:
        Absolute path as string
    """
    dir_path = Path(directory_path) if os.path.isabs(directory_path) else PROJECT_ROOT / directory_path
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)


# Common paths
DATA_DIR = get_path('data')
MODELS_DIR = get_path('models')
RESULTS_DIR = get_path('results')
NSRDB_DIR = get_path('nsrdb_data')

# Data files
X_TRAIN_FILE = get_path('data/X_train.npy')
X_TEST_FILE = get_path('data/X_test.npy')
Y_TRAIN_FILE = get_path('data/y_train.npy')
Y_TEST_FILE = get_path('data/y_test.npy')
SCALERS_FILE = get_path('data/scalers.pkl')

# Model files
LSTM_MODEL_FILE = get_path('models/lstm_model.h5')
TRANSFORMER_BEST_FILE = get_path('models/transformer_best.h5')
TRANSFORMER_MODEL_FILE = get_path('models/transformer_model.h5')

# Results files
ARIMA_PREDICTIONS_FILE = get_path('results/arima_predictions.png')
EVALUATION_REPORT_FILE = get_path('results/evaluation_report.txt')

# Spatio-Temporal Data Files
X_TRAIN_ST_FILE = get_path('data/X_train_st.npy')
X_TEST_ST_FILE = get_path('data/X_test_st.npy')
Y_TRAIN_ST_FILE = get_path('data/y_train_st.npy')
Y_TEST_ST_FILE = get_path('data/y_test_st.npy')
CORR_MATRIX_FILE = get_path('data/corr_matrix.npy')
DIST_MATRIX_FILE = get_path('data/dist_matrix.npy')
DATES_TRAIN_FILE = get_path('data/dates_train.npy')
DATES_TEST_FILE = get_path('data/dates_test.npy')
METADATA_FILE = get_path('data/metadata.pkl')

# Spatio-Temporal Model Files
TRANSFORMER_ST_BEST_FILE = get_path('models/transformer_st_best.h5')
TRANSFORMER_ST_FINAL_FILE = get_path('models/transformer_st_final.h5')
GRU_BEST_FILE = get_path('models/gru_best.h5')
GRU_FINAL_FILE = get_path('models/gru_final.h5')
