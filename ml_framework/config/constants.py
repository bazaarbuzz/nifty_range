# config/constants.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') # For plots and reports
PLOT_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'plots') # Keep plots specific

# Specific Model Type Dirs
SUPERVISED_MODEL_DIR = os.path.join(MODEL_DIR, 'supervised')
REGRESSION_MODEL_DIR = os.path.join(SUPERVISED_MODEL_DIR, 'regression') # Adjusted path
CLASSIFICATION_MODEL_DIR = os.path.join(SUPERVISED_MODEL_DIR, 'classification') # Adjusted path
UNSUPERVISED_MODEL_DIR = os.path.join(MODEL_DIR, 'unsupervised')
CLUSTERING_MODEL_DIR = os.path.join(UNSUPERVISED_MODEL_DIR, 'clustering')
DIM_REDUCTION_MODEL_DIR = os.path.join(UNSUPERVISED_MODEL_DIR, 'dim_reduction')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REGRESSION_MODEL_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)
os.makedirs(CLUSTERING_MODEL_DIR, exist_ok=True)
os.makedirs(DIM_REDUCTION_MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Project Root Directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model Directory
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REGRESSION_MODEL_DIR = os.path.join(MODEL_DIR, 'regression')
CLASSIFICATION_MODEL_DIR = os.path.join(MODEL_DIR, 'classification')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REGRESSION_MODEL_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)

# Input Data Files (assuming they are in RAW_DATA_DIR)
NIFTY_RAW_FILE = os.path.join(RAW_DATA_DIR, 'nifty_data.csv')
VIX_RAW_FILE = os.path.join(RAW_DATA_DIR, 'vix_data.csv')

CLASSIFICATION_TARGET_RANGE_BINS = 'total_range_category'
CALCULATE_MOMENTUM = True
ROLLING_WINDOWS = [3, 5, 7, 9, 11]
CALCULATE_VOLATILITY = True
CALCULATE_ROLLING_STATS = True
ADFULLER_SIGNIFICANCE_LEVEL = 0.05

# Example in config/constants.py
VALID_TARGETS = ['n_week_high', 'n_week_low', 'total_range', 'close_diff']

# Column Names
DATETIME = 'datetime'
OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'

# Derived Column Names
MON_FIRST_HOUR_PREFIX = 'mon_1h_'
EXP_PREFIX = 'exp_'
WEEK_NUM = 'week_num' # Expiry week number (1, 2, 3, 4)
MON_DATE = 'mon_date'
EXP_DATE = 'exp_date'
RANGE_HIGH = 'range_high' # High from Mon first hour start to Exp day end
RANGE_LOW = 'range_low'   # Low from Mon first hour start to Exp day end
TOTAL_RANGE = 'total_range'
MON_CLOSE_MINUS_EXP_CLOSE = 'mon_close_minus_exp_close' # Mon 1st hour close - Exp last hour close
N_WEEK_HIGH = 'n_week_high' # Alias for RANGE_HIGH
N_WEEK_LOW = 'n_week_low'   # Alias for RANGE_LOW
N_WEEK_HIGH_MINUS_MON_CLOSE = 'n_week_high_minus_mon_close'
N_WEEK_LOW_MINUS_MON_CLOSE = 'n_week_low_minus_mon_close'
N_WEEK_HIGH_MINUS_THURSDAY_CLOSE = 'n_week_high_minus_thursday_close'
N_WEEK_LOW_MINUS_THURSDAY_CLOSE = 'n_week_low_minus_thursday_close'

VIX_PREFIX_MON = 'vix_mon_'
VIX_PREFIX_EXP = 'vix_exp_'

# Timezone
TIMEZONE = 'Asia/Kolkata'

# Date/Time Formats
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z" # Matches input format

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
DEFAULT_FEATURE_SET = 'basic_ohlc_vix' # Identifier for default features used, adjust as needed

# Model Metadata Keys
MODEL_METADATA_FILE = 'model_metadata.json'
TRAINING_DATE = 'training_date'
MODEL_PARAMS = 'model_parameters'
HYPERPARAMS = 'hyperparameters'
EVAL_METRICS = 'evaluation_metrics'
FEATURE_COLS = 'feature_columns'
TARGET_COL = 'target_column'
INPUT_DATA_SOURCE = 'input_data_source'

# Evaluation Metrics Keys (examples)
METRICS_RMSE = 'RMSE'
METRICS_MAE = 'MAE'
METRICS_R2 = 'R2'
METRICS_ACCURACY = 'Accuracy'
METRICS_PRECISION = 'Precision'
METRICS_RECALL = 'Recall'
METRICS_F1 = 'F1'
METRICS_ROC_AUC = 'ROC_AUC'
METRICS_SILHOUETTE = 'Silhouette Score'
METRICS_CALINSKI = 'Calinski-Harabasz Index'
METRICS_DAVIES = 'Davies-Bouldin Index'