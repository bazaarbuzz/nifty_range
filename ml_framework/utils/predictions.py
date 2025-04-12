# utils/predictions.py

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
from datetime import datetime, timedelta # Ensure timedelta is imported
import yfinance as yf
import pytz

from config.constants import MODEL_METADATA_FILE, REGRESSION_MODEL_DIR, CLASSIFICATION_MODEL_DIR, TIMEZONE, FEATURE_COLS
from config.settings import DEFAULT_PREDICTION_DURATION_WEEKS
from utils.preprocessing import aggregate_data # Ensure this is imported

# --- load_model_and_metadata, make_prediction, get_live_data functions remain the same ---
# --- UPDATED load_model_and_metadata ---
def load_model_and_metadata(model_name, target_variable, week_num, model_type='regression'): # Added model_type
    """Loads a trained model and its metadata based on model type."""
    # Determine directory based on model_type
    if model_type == 'regression':
        model_dir = REGRESSION_MODEL_DIR
    elif model_type == 'classification':
        model_dir = CLASSIFICATION_MODEL_DIR
    # Add elif for 'clustering', 'dim_reduction' if loading those models too
    else:
        logging.error(f"Unknown model_type '{model_type}' provided.")
        # Try a default or raise error? Let's try regression as default maybe? No, better to error.
        # Try searching both? Could lead to ambiguity if names clash.
        # Let's rely on caller providing correct type.
        return None, None

    model_base_path = os.path.join(model_dir, f"{target_variable}_week{week_num}_{model_name}")
    model_path = f"{model_base_path}.joblib"
    metadata_path = f"{model_base_path}_{MODEL_METADATA_FILE}"

    model = None
    metadata = None
    logging.debug(f"Attempting to load {model_type} model from: {model_path}")

    try:
        # Load metadata first to confirm type if possible? No, model needed anyway.
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info(f"Loaded {model_type} model from {model_path}")
        else:
            logging.error(f"Model file not found: {model_path}")
            # Optional: Could try the other directory as a fallback if type was uncertain
            # if model_type == 'regression': fallback_dir = constants.CLASSIFICATION_MODEL_DIR else ... etc.
            return None, None

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata from {metadata_path}")
            # Optional: Verify metadata['model_type'] matches expected model_type
            if 'model_type' in metadata and metadata['model_type'] != model_type:
                 logging.warning(f"Metadata model_type '{metadata['model_type']}' mismatches requested type '{model_type}' for {model_name}/{target_variable}")
                 # Decide whether to proceed or return error based on mismatch tolerance
        else:
            logging.warning(f"Metadata file not found: {metadata_path}")
            metadata = {} # Return empty dict if metadata missing

    except Exception as e:
        logging.error(f"Error loading model or metadata for {model_name} ({model_type}): {e}")
        return None, metadata # Return metadata if it was loaded before model error

    return model, metadata
# --- END UPDATED load_model_and_metadata ---


# --- UPDATED make_prediction ---
def make_prediction(model, input_data, model_type='regression'): # Added model_type
    """Makes a prediction using the loaded model."""
    if model is None:
        logging.error("Model is not loaded. Cannot make prediction.")
        return None
    if not isinstance(input_data, pd.DataFrame):
         logging.error("Input data must be a pandas DataFrame.")
         return None

    try:
        if model_type == 'classification':
            # Get predicted class label
            pred_class = model.predict(input_data)
            pred_proba = None
            # Get probabilities if supported
            if hasattr(model, "predict_proba"):
                try:
                    pred_proba = model.predict_proba(input_data)
                    logging.debug(f"Predicted probabilities shape: {pred_proba.shape}")
                except Exception as proba_e:
                    logging.warning(f"Could not get predict_proba: {proba_e}")
            # Return both class and probabilities (as list/array)
            # Ensure consistent return type, maybe dict?
            return {'class': pred_class, 'proba': pred_proba}

        else: # Default to regression
            predictions = model.predict(input_data)
            # Return regression prediction (usually a single value or array)
            return {'value': predictions}

    except Exception as e:
        logging.error(f"Error during prediction ({model_type}): {e}")
        return None
# --- END UPDATED make_prediction ---

def get_live_data(nifty_ticker="^NSEI", vix_ticker="^INDIAVIX", interval="1h", period="5d"):
     """Fetches recent Nifty and VIX data using yfinance."""
     logging.info(f"Fetching live data for {nifty_ticker} and {vix_ticker}...")
     try:
         nifty = yf.Ticker(nifty_ticker)
         vix = yf.Ticker(vix_ticker)

         # Fetch hourly data for the last few days to ensure we get the latest Monday/Tuesday
         # Extend period slightly to increase chance of getting Tue if Mon is holiday
         nifty_hist = nifty.history(period=period, interval=interval)
         vix_hist = vix.history(period=period, interval=interval)

         if nifty_hist.empty: # Check nifty specifically as it drives the date logic
              logging.error("Failed to fetch sufficient live Nifty data.")
              return None, vix_hist # Return vix even if nifty failed, maybe useful context
         if vix_hist.empty:
              logging.warning("Failed to fetch live VIX data. Proceeding without VIX for prediction.")
              # VIX might be None, handle downstream

         # Rename columns to match our convention
         nifty_hist.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
         if not vix_hist.empty:
             vix_hist.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)

         # Ensure timezone is consistent
         nifty_hist.index = nifty_hist.index.tz_convert(TIMEZONE)
         if not vix_hist.empty:
            vix_hist.index = vix_hist.index.tz_convert(TIMEZONE)

         logging.info("Live data fetched successfully.")
         return nifty_hist, vix_hist

     except Exception as e:
         logging.error(f"Error fetching live data using yfinance: {e}")
         return None, None
# --- End paste ---


# --- Revised prepare_input_features function ---
def prepare_input_features(nifty_data, vix_data, feature_list):
    """
    Prepares the single row DataFrame for prediction based on the latest available
    trading day's first hour data, attempting Mon -> Tue fallback if Monday's
    first hour is missing.
    Uses the aggregate_data helper from preprocessing for consistency.
    """
    logging.info("Preparing input features from live data...")
    from config.settings import FIRST_HOUR_START, FIRST_HOUR_END
    from config.constants import (MON_FIRST_HOUR_PREFIX, VIX_PREFIX_MON, OPEN, HIGH, LOW, CLOSE)

    # --- 1. Ensure Nifty data has required columns ---
    if nifty_data is None or nifty_data.empty:
         logging.error("Nifty data is empty, cannot prepare features.")
         return None
    if 'date' not in nifty_data.columns: nifty_data['date'] = nifty_data.index.date
    if 'time' not in nifty_data.columns: nifty_data['time'] = nifty_data.index.time
    if 'weekday' not in nifty_data.columns: nifty_data['weekday'] = nifty_data.index.weekday

    # --- 2. Find the latest date available in Nifty data ---
    if nifty_data.index.empty:
        logging.error("Nifty data index is empty after initial processing.")
        return None
    latest_available_date = nifty_data.index.date.max()
    latest_available_weekday = pd.Timestamp(latest_available_date).weekday() # Monday=0, Sunday=6
    logging.info(f"Latest available date in fetched Nifty data: {latest_available_date} (Weekday: {latest_available_weekday})")

    # --- 3. Determine the target start date ---
    # Our primary target is Monday. If the latest available day IS Monday, we use it.
    # If the latest available day is TUE/WED/etc., we assume Monday was holiday/missing
    # and use the latest available day as the effective "start" day.
    target_start_date = latest_available_date
    is_latest_day_monday = (latest_available_weekday == 0)
    logging.info(f"Using {target_start_date} as the initial target start day.")

    # --- 4. Try getting data for the target start date ---
    first_hour_start_time = datetime.strptime(FIRST_HOUR_START, '%H:%M').time()
    first_hour_end_time = datetime.strptime(FIRST_HOUR_END, '%H:%M').time()
    actual_start_date_used = None # Store the date we actually use data from
    nifty_first_hour_data = None # Store the resulting data slice

    start_day_first_hour_nifty = nifty_data[
        (nifty_data['date'] == target_start_date) &
        (nifty_data.index.time >= first_hour_start_time) &
        (nifty_data.index.time < first_hour_end_time)
    ]

    if not start_day_first_hour_nifty.empty:
        logging.info(f"Found first hour data for target start day: {target_start_date}")
        actual_start_date_used = target_start_date
        nifty_first_hour_data = start_day_first_hour_nifty # Use this data
    elif is_latest_day_monday: # Only try fallback to Tue if the latest available day was Mon AND its data was missing
        # --- 5. Latest was Monday, data missing, try Tuesday fallback ---
        logging.warning(f"No first hour Nifty data found for the latest Monday {target_start_date}. Trying next day (Tuesday).")
        # Calculate the date for the next day
        target_tuesday = target_start_date + timedelta(days=1)
        # Check if this Tuesday exists in the fetched data dates
        available_dates = nifty_data['date'].unique()
        if target_tuesday not in available_dates:
             logging.error(f"Fallback Tuesday {target_tuesday} not found in fetched Nifty data range ({available_dates.min()} to {available_dates.max()}). Cannot find suitable start day.")
             return None

        tue_first_hour_nifty = nifty_data[
            (nifty_data['date'] == target_tuesday) &
            (nifty_data.index.time >= first_hour_start_time) &
            (nifty_data.index.time < first_hour_end_time)
        ]

        if not tue_first_hour_nifty.empty:
            logging.info(f"Found first hour data for fallback Tuesday: {target_tuesday}. Using this as start day.")
            actual_start_date_used = target_tuesday
            nifty_first_hour_data = tue_first_hour_nifty # Use Tuesday's data
        else:
            logging.error(f"No first hour Nifty data found for fallback Tuesday {target_tuesday} either.")
            return None # Failed to find start day data
    else: # Latest available day wasn't Monday, and its first hour data is missing. Fail.
         logging.error(f"No first hour Nifty data found for the latest available day {target_start_date} (which was not Monday).")
         return None # Failed to find start day data

    # --- 6. Aggregate Nifty data (from Mon OR Tue OR latest day) ---
    if nifty_first_hour_data is None or nifty_first_hour_data.empty:
         logging.error("Internal error: Nifty data slice for aggregation is unexpectedly empty.")
         return None
    mon_1h_agg_nifty = aggregate_data(nifty_first_hour_data, MON_FIRST_HOUR_PREFIX)

    # --- 7. Get VIX data for the *actual* start date used ---
    vix_mon_agg = pd.Series(dtype=float) # Initialize empty series
    if vix_data is not None and not vix_data.empty:
        if 'date' not in vix_data.columns: vix_data['date'] = vix_data.index.date
        if 'time' not in vix_data.columns: vix_data['time'] = vix_data.index.time

        mon_first_hour_vix = vix_data[
            (vix_data['date'] == actual_start_date_used) &
            (vix_data.index.time >= first_hour_start_time) &
            (vix_data.index.time < first_hour_end_time)
        ]
        vix_mon_agg = aggregate_data(mon_first_hour_vix, VIX_PREFIX_MON)
        if mon_first_hour_vix.empty:
             logging.warning(f"No first hour VIX data found for actual start date {actual_start_date_used}. Using NaNs.")
    else:
        logging.warning("VIX data was not available or empty. Proceeding without VIX features.")
        # Need to create NaN placeholders for VIX columns expected by the model
        vix_cols = [col for col in feature_list if col.startswith(VIX_PREFIX_MON)]
        vix_mon_agg = pd.Series(np.nan, index=vix_cols)


    # --- 8. Combine features ---
    # Ensure indices don't clash if vix_mon_agg was created manually
    vix_mon_agg.index.name = None
    mon_1h_agg_nifty.index.name = None
    input_features = pd.concat([mon_1h_agg_nifty, vix_mon_agg])
    input_df = pd.DataFrame([input_features])

    # --- 9. Ensure all expected columns are present ---
    # Add missing columns expected by the model but not generated (e.g., if VIX failed)
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = np.nan
            logging.warning(f"Feature '{col}' missing after aggregation/concat, adding as NaN.")

    # --- 10. Reorder columns ---
    try:
        # Only select columns that are actually in the feature_list
        cols_to_select = [col for col in feature_list if col in input_df.columns]
        # Check if all required features are present now
        missing_model_features = [col for col in feature_list if col not in cols_to_select]
        if missing_model_features:
             # This check is slightly redundant if the loop above works, but good safeguard
             logging.error(f"Model requires features not found/generated: {missing_model_features}. Prediction may fail or be inaccurate.")

        input_df = input_df[feature_list] # Select exactly the features in order
    except KeyError as e:
         logging.error(f"Error reordering columns. Attempted to select: {feature_list}. Available: {input_df.columns.tolist()}. Error: {e}")
         return None
    except Exception as e:
         logging.error(f"Unexpected error during column reordering: {e}")
         return None


    logging.info(f"Input features prepared using data from: {actual_start_date_used}.")
    # Check for NaNs in critical columns
    critical_cols_present = [col for col in FEATURE_COLS if col in input_df.columns] # Check only cols that exist
    if not critical_cols_present:
         logging.warning("No critical columns found for NaN check.")
    elif input_df[critical_cols_present].isnull().any().any():
         logging.warning(f"NaN values detected in critical input features after preparation: \n{input_df[critical_cols_present].isnull().sum()[input_df[critical_cols_present].isnull().sum() > 0]}")

    return input_df

# --- format_prediction_output function (ensure the latest version is pasted here) ---
def format_prediction_output(predictions, metadata, input_info):
    """Formats the prediction results into a JSON structure."""
    output = {
        "prediction_timestamp": datetime.now(pytz.timezone(TIMEZONE)).isoformat(),
        "input_data_info": input_info,
        "model_metadata": metadata,
        "predictions": {} # Default empty dict
    }

    if isinstance(predictions, dict):
         output["predictions"] = predictions
    else:
         logging.warning("Predictions passed to format_prediction_output are not a dict. Formatting as single value.")
         output["predictions"]["prediction"] = predictions

    try:
        # Updated default handler for common types
        def default_serializer(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                # Handle NaN and Inf specifically for JSON compatibility
                if np.isnan(o): return None # Represent NaN as null
                if np.isinf(o): return 'Infinity' if o > 0 else '-Infinity'
                return float(o)
            elif isinstance(o, (np.complex_, np.complex64, np.complex128)):
                return {'real': o.real, 'imag': o.imag}
            elif isinstance(o, (np.ndarray,)):
                # Recursively apply serializer to list elements if needed, handle NaN/Inf
                return [default_serializer(item) for item in o]
            elif isinstance(o, (np.bool_)):
                return bool(o)
            elif isinstance(o, (np.void)):
                return None
            elif isinstance(o, (datetime, pd.Timestamp)):
                return o.isoformat()
            elif isinstance(o, pd.Timedelta):
                 return str(o)
            # Let the base default method raise the TypeError for unsupported types
            # return json.JSONEncoder.default(self, o) # Not needed if just raising
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


        json_output = json.dumps(output, indent=4, default=default_serializer)
        return json_output
    except TypeError as e:
        logging.error(f"Error serializing prediction output to JSON: {e}")
        # Minimal fallback
        return json.dumps({"error": "Failed to serialize prediction output", "details": str(e)}, indent=4)

# --- main() and if __name__ == "__main__": blocks remain the same as the previous multi-target version ---
# (Ensure they are present below this line)
# ... (paste the main function and argparse setup from the previous correct version) ...