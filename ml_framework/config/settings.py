# config/settings.py
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from config.constants import (
    TOTAL_RANGE, MON_CLOSE_MINUS_EXP_CLOSE, N_WEEK_HIGH, N_WEEK_LOW,
    N_WEEK_HIGH_MINUS_MON_CLOSE, N_WEEK_LOW_MINUS_MON_CLOSE,
    N_WEEK_HIGH_MINUS_THURSDAY_CLOSE, N_WEEK_LOW_MINUS_THURSDAY_CLOSE,
    MON_FIRST_HOUR_PREFIX, OPEN, HIGH, LOW, CLOSE, VIX_PREFIX_MON, VIX_PREFIX_EXP, CLASSIFICATION_TARGET_RANGE_BINS,
    METRICS_RMSE, METRICS_MAE, METRICS_R2, METRICS_ACCURACY, METRICS_PRECISION, METRICS_RECALL, METRICS_F1, METRICS_ROC_AUC, METRICS_SILHOUETTE, METRICS_CALINSKI, METRICS_DAVIES
)

# --- Data Processing Settings ---
FIRST_HOUR_START = "09:15"
FIRST_HOUR_END = "10:15" # Exclusive end time for filtering
LAST_HOUR_START = "14:30" # Example, adjust based on actual market close hour for Nifty/Expiry
LAST_HOUR_END = "15:30"   # Example

# --- Model Training Settings ---

# Define target variables for regression
REGRESSION_TARGETS = [
    TOTAL_RANGE,
    MON_CLOSE_MINUS_EXP_CLOSE,
    N_WEEK_HIGH,
    N_WEEK_LOW,
    N_WEEK_HIGH_MINUS_MON_CLOSE,
    N_WEEK_LOW_MINUS_MON_CLOSE,
    N_WEEK_HIGH_MINUS_THURSDAY_CLOSE,
    N_WEEK_LOW_MINUS_THURSDAY_CLOSE,
]

# Define potential features (adjust based on analysis)
# These are the base names, prefixes like 'mon_1h_', 'vix_mon_', 'vix_exp_' will be added
BASE_FEATURE_COLS = [OPEN, HIGH, LOW, CLOSE] # From Nifty Monday 1st Hour
VIX_FEATURE_COLS = [OPEN, HIGH, LOW, CLOSE] # From VIX Monday and Expiry

CLASSIFICATION_TARGET_RANGE_BINS = 'total_range_category'
AVAILABLE_CLASSIFICATION_TARGETS = [CLASSIFICATION_TARGET_RANGE_BINS]
AVAILABLE_REGRESSION_TARGETS = REGRESSION_TARGETS
# Define engineered features (example)  

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_json')


# Construct full feature list (example)
FEATURE_COLUMNS = \
    [f"{MON_FIRST_HOUR_PREFIX}{col}" for col in BASE_FEATURE_COLS] + \
    [f"{VIX_PREFIX_MON}{col}" for col in VIX_FEATURE_COLS] + \
    [f"{VIX_PREFIX_EXP}{col}" for col in VIX_FEATURE_COLS]
    # Add more engineered features if needed

# Define models to use
REGRESSION_MODELS = {
    'ElasticNet': ElasticNet(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'XGBRegressor': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    'LGBMRegressor': lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    'CatBoostRegressor': cb.CatBoostRegressor(random_state=42, verbose=0) # verbose=0 to reduce noise
}

CLASSIFICATION_MODELS = {
    # Define classification models if needed (e.g., predicting direction Up/Down)
    # 'LogisticRegression': LogisticRegression(random_state=42),
    # 'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1),
    # 'XGBClassifier': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), # Specify eval_metric
    # 'CatBoostClassifier': cb.CatBoostClassifier(random_state=42, verbose=0)
}

# List models available in the respective model files
AVAILABLE_REGRESSION_MODELS = ['ElasticNet', 'RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']
AVAILABLE_CLASSIFICATION_MODELS = ['LogisticRegression', 'SVC', 'KNeighborsClassifier', 'RandomForestClassifier', 'CatBoostClassifier', 'XGBClassifier'] # Add NaiveBayes?
AVAILABLE_CLUSTERING_MODELS = ['KMeans', 'GaussianMixture', 'DBSCAN', 'IsolationForest']
AVAILABLE_DIM_REDUCTION_MODELS = ['PCA', 'UMAP', 'TSNE']

# Basic Hyperparameter Grids (Example - Expand significantly for real tuning)
# Use smaller grids for faster demonstration
HYPERPARAM_GRIDS = {
    'ElasticNet': {'alpha': [0.1, 0.5, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]},
    'RandomForestRegressor': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
    'XGBRegressor': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    'LGBMRegressor': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 31]},
    'CatBoostRegressor': {'iterations': [50, 100], 'learning_rate': [0.05, 0.1], 'depth': [4, 6]},
}

# --- Training & Evaluation Settings ---
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5 # For evaluation and tuning
TUNING_METHOD = 'GridSearchCV' # or 'RandomizedSearchCV'
TUNING_ITERATIONS = 50 # For RandomizedSearchCV
DEFAULT_FEATURE_SET = 'basic_ohlc_vix' # Identifier for default features used, adjust as needed

# Metrics for evaluation reports (keys match constants.py)
DEFAULT_REGRESSION_METRICS = [METRICS_RMSE, METRICS_MAE, METRICS_R2]
DEFAULT_CLASSIFICATION_METRICS = [METRICS_ACCURACY, METRICS_PRECISION, METRICS_RECALL, METRICS_F1, METRICS_ROC_AUC]
DEFAULT_CLUSTERING_METRICS = [METRICS_SILHOUETTE, METRICS_CALINSKI, METRICS_DAVIES]

# --- Prediction Settings ---
DEFAULT_PREDICTION_DURATION_WEEKS = 3 # For predict.py auto mode
ENSEMBLE_METHOD = 'average' # 'average', 'median', 'weighted' (requires weights)
# Confidence Interval calculation method (e.g., 'bootstrap', 'quantile', None)
CONFIDENCE_INTERVAL_METHOD = None # Keep simple for now
CONFIDENCE_LEVEL = 0.95 # e.g., for 95% CI

# Metrics for evaluation
REGRESSION_METRICS = ['neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
CLASSIFICATION_METRICS = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc'] # Example
AVAILABLE_CLASSIFICATION_TARGETS = [
    # constants.CLASSIFICATION_TARGET_RANGE_BINS, # If you have this one
    'price_direction'  # <------ ADD OR ENSURE THIS LINE EXISTS EXACTLY
]
# --- Backtesting Settings ---
BACKTEST_CONSECUTIVE_FAILURE_THRESHOLD = 3

# --- Prediction Settings ---
DEFAULT_PREDICTION_DURATION_WEEKS = 3 # For predict.py auto mode

# --- Visualization Settings ---
PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
FIG_SIZE = (12, 6)
CORR_MATRIX_FIG_SIZE = (10, 8)
FEATURE_IMPORTANCE_TOP_N = 15
CLUSTER_PLOT_DIMENSIONS = 2 # 2 or 3