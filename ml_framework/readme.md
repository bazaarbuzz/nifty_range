Okay, here is a comprehensive README.md file tailored to your project structure and the command-line arguments we've implemented.

# Nifty/VIX Weekly Expiry ML Framework

This project provides a modular, reusable, and configurable Python-based machine learning framework designed to predict Nifty index weekly expiry characteristics (like high-low range) based on Nifty and India VIX data. It automates data preprocessing, model training, evaluation, visualization, and provides prediction capabilities via command-line interfaces.

## Features

*   **Modular Structure:** Code is organized into distinct modules (config, data, models, utils) for maintainability and scalability.
*   **Configurable:** Easily configure models, hyperparameters, features, and file paths through settings files.
*   **Automated Preprocessing:** Processes raw hourly Nifty (OHLC) and VIX (OHLC) data into weekly expiry feature sets, handling potential missing trading days (Mon/Thu holidays).
*   **Automated Training & Evaluation:** Trains multiple supervised learning models (Regression focus initially) for specified target variables and expiry weeks. Includes hyperparameter tuning (optional) and evaluation metric calculation.
*   **Automated Visualization:** Generates plots for data analysis (correlations, distributions) and model evaluation (predictions vs actual, feature importance).
*   **Command-Line Driven:** All core functionalities (preprocessing, analysis, training, prediction) are accessible via `main.py` and `predict.py` using command-line arguments.
*   **Prediction System:** Offers modes for prediction using manually entered features or automatically fetched live data (via `yfinance`).

## Folder Structure


ml_framework/
├── config/
│ ├── constants.py # Global constants (paths, column names)
│ └── settings.py # Configurable settings (models, targets, tuning)
├── data/
│ ├── raw/ # Place input CSV files here (nifty_data.csv, vix_data.csv)
│ └── processed/ # Output of preprocessing step
├── models/
│ ├── regression/ # Saved regression model files (.joblib, .json)
│ └── classification/ # (Placeholder for future classification models)
├── notebooks/ # Jupyter notebooks for exploration/debugging
├── results/
│ └── plots/ # Saved plots from analysis and evaluation
├── utils/
│ ├── preprocessing.py # Data loading and feature engineering logic
│ ├── visualization.py # Plotting functions
│ └── predictions.py # Prediction logic (loading models, inference)
├── main.py # Entry point for preprocessing, analysis, training
├── predict.py # Entry point for making predictions
├── requirements.txt # Project dependencies
├── README.md # This file
└── .gitignore # Git ignore rules

## Prerequisites

*   Python 3.9+
*   pip

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ml_framework
    ```

2.  **Create and activate a virtual environment (Recommended):**
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    *   Place your raw hourly Nifty data file named `nifty_data.csv` into the `data/raw/` folder. It **must** contain columns: `datetime`, `open`, `high`, `low`, `close`.
    *   Place your raw hourly India VIX data file named `vix_data.csv` into the `data/raw/` folder. It **must** contain columns: `datetime`, `open`, `high`, `low`, `close`.
    *   Ensure the `datetime` column format matches `YYYY-MM-DDTHH:MM:SS+HH:MM` (e.g., `2014-08-12T10:15:00+05:30`) or adjust `DATETIME_FORMAT` in `config/constants.py`.

## Usage

The framework is primarily operated through `main.py` (for data preparation and model training) and `predict.py` (for generating predictions).

### 1. `main.py` - Preprocessing, Analysis, Training

This script orchestrates the workflow from raw data to trained models.

**Command Structure:**

```bash
python main.py [actions] [options]


Actions (Specify at least one):

--preprocess: Loads raw data from data/raw/, performs feature engineering for weekly expiries (handling Mon/Thu holidays by shifting to Tue/Wed), and saves processed files to data/processed/.

--analyze: Loads processed data (requires --preprocess to be run previously or concurrently) and generates analysis plots (distributions, correlations) saved in results/plots/.

--train: Loads processed data, splits it, trains specified models for specified targets/weeks, performs hyperparameter tuning (optional), evaluates models, and saves models (.joblib), metadata (.json), and evaluation plots (.png) to models/regression/ and results/plots/.

Options:

--expiry_week <N...>: Specify which expiry week(s) to process (1, 2, 3, or 4). Default is all weeks (1 2 3 4).

Example: --expiry_week 1 2 (Process weeks 1 and 2)

--target <name...>: Specify which target variable(s) from config/settings.py (REGRESSION_TARGETS) to train models for. Default is all defined targets.

Example: --target total_range mon_close_minus_exp_close

--model <Name...>: Specify which model(s) defined in config/settings.py (REGRESSION_MODELS) to train. Default is all defined models. Model names must match keys in the settings dictionary (e.g., XGBRegressor, RandomForestRegressor).

Example: --model XGBRegressor ElasticNet

--no_tune: If included with --train, skips hyperparameter tuning (GridSearchCV) and trains models with their default parameters as defined in config/settings.py.

Examples:

Run only preprocessing for all weeks:

python main.py --preprocess


Preprocess and Analyze data for Week 1 expiry:

python main.py --preprocess --analyze --expiry_week 1


Train default models for default targets for Weeks 1 and 2 (assumes preprocessing was done):

python main.py --train --expiry_week 1 2


Preprocess, Analyze, and Train all default models/targets for all weeks:

python main.py --preprocess --analyze --train


Train only XGBoost and Random Forest for the total_range target for Week 1 expiry, without hyperparameter tuning:

python main.py --train --expiry_week 1 --target total_range --model XGBRegressor RandomForestRegressor --no_tune

2. predict.py - Making Predictions

This script uses previously trained models to generate predictions.

Command Structure:

python predict.py --mode <mode> --model_name <name> --target <target> --week_num <N> [mode_specific_args...]

Required Arguments:

--mode <auto|manual>: Choose the input mode:

auto: Fetch latest Nifty/VIX data using yfinance to create input features.

manual: Use feature values provided directly via command-line arguments.

--model_name <Name>: The name of the trained model to use (must match a saved model prefix, e.g., XGBRegressor).

--target <name>: The target variable the specified model was trained to predict (e.g., total_range).

--week_num <N>: The expiry week (1-4) the specified model was trained for.

Mode-Specific Arguments:

For --mode auto:

--nifty_ticker <symbol>: (Optional) yfinance ticker for Nifty. Default: ^NSEI.

--vix_ticker <symbol>: (Optional) yfinance ticker for VIX. Default: ^INDIAVIX.

--interval <interval>: (Optional) Data interval for yfinance fetch. Default: 1h.

--period <period>: (Optional) Data period for yfinance fetch. Default: 5d.

For --mode manual: Provide the required feature values corresponding to Monday's (or Tuesday's if Mon holiday) first hour data. Minimally, --nifty_close and --vix_close are often required, but provide all features the model was trained on for best results:

--nifty_open <value>

--nifty_high <value>

--nifty_low <value>

--nifty_close <value>

--vix_open <value>

--vix_high <value>

--vix_low <value>

--vix_close <value>

(Add other features like VIX expiry values if your models use them)

Examples:

Predict total_range for Week 1 using a trained XGBRegressor model, fetching live data automatically:

python predict.py --mode auto --model_names XGBRegressor RandomForestRegressor --target n_week_high --week_num 2


Predict mon_close_minus_exp_close for Week 2 using a trained RandomForestRegressor model, providing manual inputs for Monday's first hour:

python predict.py --mode manual --model_names RandomForestRegressor XGBRegressor LGBMRegressor --target n_week_high --week_num 2 \
    --nifty_open 21758.40 --nifty_high 22188 --nifty_low 21758 --nifty_close 21918.15 \
    --vix_open 18.66 --vix_high 21.94 --vix_low 17.07 --vix_close 21.23
Output:

predict.py prints a JSON object to the standard output containing:

Prediction timestamp

Input data information (mode, tickers/manual values used)

Metadata of the loaded model (training date, params, metrics, features)

The predicted value(s)

Outputs

Running the scripts generates the following:

Processed Data: CSV files in data/processed/ (e.g., week1_expiry_features.csv).

Analysis & Evaluation Plots: PNG images in results/plots/.

Trained Models: Joblib files in models/regression/ (e.g., total_range_week1_XGBRegressor.joblib).

Model Metadata: JSON files alongside models in models/regression/ (e.g., total_range_week1_XGBRegressor_model_metadata.json).

Predictions: JSON output to console from predict.py.

Configuration

Models, Targets, Tuning: Modify config/settings.py to add/remove models, change target variables, adjust hyperparameter grids (HYPERPARAM_GRIDS), set test split ratio, CV folds, etc.

Paths, Column Names, Constants: Modify config/constants.py to change directory paths, input filenames, or internal column naming conventions.

Notebooks

The notebooks/ directory contains Jupyter notebooks for interactive data exploration, analysis, and debugging purposes.

Save this content as `README.md` in the root directory of your `ml_framework` project. Remember to replace `<your-repository-url>` if you plan to host this on Git.

# instead of --target assume all targets are needed, loop for all the target and collate the responses
# given a series of dates in a list, get the ohlc of nifty & vix from the local csv files and get the prediction
# do the testing on model from 2021, and we should have the results of predictions(number of correct range bounded prediction, number of incorrect range bounded prediction, number of correct out of range prediction, number of incorrect out of range prediction)
# classification models to be added for the weekly range and calculating 
# 3 consecutive failures check
# 