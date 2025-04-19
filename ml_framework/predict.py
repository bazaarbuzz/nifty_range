# predict.py
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta # Ensure timedelta is imported
import pytz
import yfinance as yf
# Assuming constants and settings are correctly configured
from config import constants, settings
from utils import predictions as pred_utils # Keep separate for clarity if desired, or merge functions here
from utils.preprocessing import aggregate_data # Use the helper

# Use constants for logging config
logging.basicConfig(level=constants.LOG_LEVEL, format=constants.LOG_FORMAT)

# ==========================================
# Utility Functions (Could live in utils/predictions.py)
# ==========================================

def load_model_and_metadata(model_name, target_variable, week_num, model_type='regression'): # Added model_type
    """Loads a trained model and its metadata based on model type."""
    # Determine directory based on model_type
    if model_type == 'regression':
        model_dir = constants.REGRESSION_MODEL_DIR
    elif model_type == 'classification':
        model_dir = constants.CLASSIFICATION_MODEL_DIR
    # Add elif for 'clustering', 'dim_reduction' if loading those models too
    else:
        logging.error(f"Unknown model_type '{model_type}' provided for loading.")
        return None, None

    model_base_path = os.path.join(model_dir, f"{target_variable}_week{week_num}_{model_name}")
    model_path = f"{model_base_path}.joblib"
    metadata_path = f"{model_base_path}_{constants.MODEL_METADATA_FILE}"

    model = None
    metadata = None
    logging.debug(f"Attempting to load {model_type} model from: {model_path}")

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info(f"Loaded {model_type} model from {model_path}")
        else:
            logging.error(f"Model file not found: {model_path}")
            return None, None

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata from {metadata_path}")
            # Optional: Verify metadata type matches expected type
            if 'model_type' in metadata and metadata['model_type'] != model_type:
                 logging.warning(f"Metadata model_type '{metadata['model_type']}' mismatches requested type '{model_type}' for {model_name}/{target_variable}")
        else:
            logging.warning(f"Metadata file not found: {metadata_path}")
            metadata = {} # Return empty dict if metadata missing

    except Exception as e:
        logging.error(f"Error loading model or metadata for {model_name} ({model_type}): {e}")
        return None, metadata # Return metadata if loaded before model error

    return model, metadata

def make_prediction(model, input_data, model_type='regression'): # Added model_type
    """Makes a prediction using the loaded model, returning a structured dict."""
    if model is None:
        logging.error("Model is not loaded. Cannot make prediction.")
        return None
    if not isinstance(input_data, pd.DataFrame):
         logging.error("Input data must be a pandas DataFrame.")
         return None

    try:
        if model_type == 'classification':
            pred_class = model.predict(input_data)
            pred_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    pred_proba = model.predict_proba(input_data)
                    logging.debug(f"Predicted probabilities shape: {pred_proba.shape}")
                except Exception as proba_e:
                    logging.warning(f"Could not get predict_proba: {proba_e}")
            # Return both class and probabilities
            return {'class': pred_class, 'proba': pred_proba}

        else: # Default to regression
            predictions = model.predict(input_data)
            # Return regression prediction
            return {'value': predictions}

    except Exception as e:
        logging.error(f"Error during prediction ({model_type}): {e}")
        return None

def get_live_data(nifty_ticker="^NSEI", vix_ticker="^INDIAVIX", interval="1h", period="7d"): # Increased period slightly
     """Fetches recent Nifty and VIX data using yfinance."""
     logging.info(f"Fetching live data for {nifty_ticker} and {vix_ticker}...")
     try:
         nifty = yf.Ticker(nifty_ticker)
         vix = yf.Ticker(vix_ticker)

         nifty_hist = nifty.history(period=period, interval=interval)
         vix_hist = vix.history(period=period, interval=interval)

         if nifty_hist.empty:
              logging.error("Failed to fetch sufficient live Nifty data.")
              return None, vix_hist # Return vix if available
         if vix_hist.empty:
              logging.warning("Failed to fetch live VIX data. Proceeding without VIX for prediction.")
              # Handle None VIX downstream

         # Rename columns
         nifty_hist.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
         if not vix_hist.empty:
             vix_hist.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)

         # Ensure timezone
         nifty_hist.index = nifty_hist.index.tz_convert(constants.TIMEZONE)
         if not vix_hist.empty:
            vix_hist.index = vix_hist.index.tz_convert(constants.TIMEZONE)

         logging.info("Live data fetched successfully.")
         return nifty_hist, vix_hist

     except Exception as e:
         logging.error(f"Error fetching live data using yfinance: {e}")
         return None, None

def prepare_input_features(nifty_data, vix_data, feature_list):
    """
    Prepares the single row DataFrame for prediction based on the latest available
    trading day's first hour data, attempting Mon -> Tue fallback if Monday's
    first hour is missing. Uses the aggregate_data helper.
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
    latest_available_weekday = pd.Timestamp(latest_available_date).weekday() # Monday=0
    logging.info(f"Latest available date in fetched Nifty data: {latest_available_date} (Weekday: {latest_available_weekday})")

    # --- 3. Determine target start date & Try getting its data ---
    target_start_date = latest_available_date
    is_latest_day_monday = (latest_available_weekday == 0)
    logging.info(f"Using {target_start_date} as the initial target start day.")

    first_hour_start_time = datetime.strptime(FIRST_HOUR_START, '%H:%M').time()
    first_hour_end_time = datetime.strptime(FIRST_HOUR_END, '%H:%M').time()
    actual_start_date_used = None
    nifty_first_hour_data = None

    start_day_first_hour_nifty = nifty_data[
        (nifty_data['date'] == target_start_date) &
        (nifty_data.index.time >= first_hour_start_time) &
        (nifty_data.index.time < first_hour_end_time)
    ]

    if not start_day_first_hour_nifty.empty:
        logging.info(f"Found first hour data for target start day: {target_start_date}")
        actual_start_date_used = target_start_date
        nifty_first_hour_data = start_day_first_hour_nifty
    elif is_latest_day_monday: # Only try fallback if latest available was Mon AND its data was missing
        logging.warning(f"No first hour Nifty data found for the latest Monday {target_start_date}. Trying next day (Tuesday).")
        target_tuesday = target_start_date + timedelta(days=1)
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
            nifty_first_hour_data = tue_first_hour_nifty
        else:
            logging.error(f"No first hour Nifty data found for fallback Tuesday {target_tuesday} either.")
            return None
    else: # Latest wasn't Monday, and its data is missing. Fail.
         logging.error(f"No first hour Nifty data found for the latest available day {target_start_date} (which was not Monday).")
         return None

    # --- 4. Aggregate Nifty data ---
    if nifty_first_hour_data is None or nifty_first_hour_data.empty:
         logging.error("Internal error: Nifty data slice for aggregation is unexpectedly empty.")
         return None
    mon_1h_agg_nifty = aggregate_data(nifty_first_hour_data, constants.MON_FIRST_HOUR_PREFIX)

    # --- 5. Get & Aggregate VIX data for the actual start date ---
    vix_mon_agg = pd.Series(dtype=float)
    if vix_data is not None and not vix_data.empty:
        if 'date' not in vix_data.columns: vix_data['date'] = vix_data.index.date
        if 'time' not in vix_data.columns: vix_data['time'] = vix_data.index.time

        mon_first_hour_vix = vix_data[
            (vix_data['date'] == actual_start_date_used) &
            (vix_data.index.time >= first_hour_start_time) &
            (vix_data.index.time < first_hour_end_time)
        ]
        vix_mon_agg = aggregate_data(mon_first_hour_vix, constants.VIX_PREFIX_MON)
        if mon_first_hour_vix.empty:
             logging.warning(f"No first hour VIX data found for actual start date {actual_start_date_used}. Using NaNs.")
    else:
        logging.warning("VIX data was not available or empty. Proceeding without VIX features.")
        vix_cols = [col for col in feature_list if col.startswith(constants.VIX_PREFIX_MON)]
        vix_mon_agg = pd.Series(np.nan, index=vix_cols) # Create NaN placeholders

    # --- 6. Combine features ---
    vix_mon_agg.index.name = None
    mon_1h_agg_nifty.index.name = None
    input_features = pd.concat([mon_1h_agg_nifty, vix_mon_agg])
    input_df = pd.DataFrame([input_features])

    # --- 7. Ensure all expected columns are present & Reorder---
    for col in feature_list:
        if col not in input_df.columns:
            input_df[col] = np.nan
            logging.warning(f"Feature '{col}' missing after aggregation/concat, adding as NaN.")
    try:
        input_df = input_df[feature_list] # Select exactly the features in order
    except KeyError as e:
         missing_in_df = [f for f in feature_list if f not in input_df.columns]
         logging.error(f"Error reordering columns. Model expects: {feature_list}. Available after prep: {input_df.columns.tolist()}. Missing: {missing_in_df}. Error: {e}")
         return None
    except Exception as e:
         logging.error(f"Unexpected error during column reordering: {e}")
         return None

    logging.info(f"Input features prepared using data from: {actual_start_date_used}.")
    # Optional: Check for NaNs in critical columns before returning
    # ... (NaN check code as before) ...

    return input_df

def format_prediction_output(predictions, metadata, input_info):
    """Formats the prediction results into a JSON structure with robust serialization."""
    output = {
        "prediction_timestamp": datetime.now(pytz.timezone(constants.TIMEZONE)).isoformat(),
        "input_data_info": input_info,
        "model_metadata": metadata, # Metadata passed might be structured (e.g., best model per target)
        "predictions": predictions # Pass the prediction dictionary directly
    }

    try:
        # Comprehensive default serializer
        def default_serializer(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(o)
            if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                if np.isnan(o): return None
                if np.isinf(o): return 'Infinity' if o > 0 else '-Infinity'
                return float(o)
            if isinstance(o, (np.complex_, np.complex64, np.complex128)): return {'real': o.real, 'imag': o.imag}
            if isinstance(o, (np.ndarray,)): return [default_serializer(item) for item in o] # Handle arrays recursively
            if isinstance(o, (np.bool_)): return bool(o)
            if isinstance(o, (np.void)): return None
            if isinstance(o, (datetime, pd.Timestamp)): return o.isoformat()
            if isinstance(o, pd.Timedelta): return str(o)
            if isinstance(o, (pd.Period)): return str(o)
            # Handle sets by converting to list
            if isinstance(o, set): return list(o)
            # Let the base default method raise the TypeError for unsupported types
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        json_output = json.dumps(output, indent=4, default=default_serializer)
        return json_output
    except TypeError as e:
        logging.error(f"Error serializing prediction output to JSON: {e}")
        # Minimal fallback
        return json.dumps({"error": "Failed to serialize prediction output", "details": str(e)}, indent=4)

# ==========================================
# Prediction Logic Function
# ==========================================

def predict_from_features(feature_df, model_name, target_variable, week_num, model_type='regression'):
    """Loads model (based on type) and predicts using provided feature DataFrame."""
    if feature_df is None or feature_df.empty:
        logging.error(f"[{target_variable} - {model_name}] Feature DataFrame is empty. Cannot predict.")
        return None, None

    feature_df_copy = feature_df.copy()
    # Load model using the specified type to search correct directory
    model, metadata = load_model_and_metadata(model_name, target_variable, week_num, model_type)

    if model is None:
        logging.error(f"[{target_variable} - {model_name}] Failed to load model.")
        return None, metadata
    # Metadata check happens within load_model_and_metadata now

    # Feature checking logic (remains same, uses metadata)
    model_features = metadata.get(constants.FEATURE_COLS, []) if metadata else []
    if not model_features:
         logging.warning(f"[{target_variable} - {model_name}] Model feature list not found in metadata. Proceeding without check.")
         model_features = feature_df_copy.columns.tolist() # Assume input is correct

    missing_cols = [col for col in model_features if col not in feature_df_copy.columns]
    if missing_cols:
        logging.error(f"[{target_variable} - {model_name}] Input features missing columns required by model: {missing_cols}. Cannot predict.")
        return None, metadata

    try:
        feature_df_copy = feature_df_copy[model_features]
    except KeyError as e:
        logging.error(f"[{target_variable} - {model_name}] Error selecting/reordering columns: {e}.")
        return None, metadata

    # Make prediction using the correct type
    # make_prediction now returns a dict: {'value': ...} or {'class': ..., 'proba': ...}
    prediction_dict = make_prediction(model, feature_df_copy, model_type=model_type)

    if prediction_dict is None:
        logging.error(f"[{target_variable} - {model_name}] Prediction failed using the model.")
        return None, metadata

    # Extract relevant result for logging/return
    if model_type == 'classification':
         pred_class = prediction_dict.get('class')
         pred_proba = prediction_dict.get('proba')
         # Safely get first element if it's array/list
         log_class = pred_class[0] if isinstance(pred_class, (np.ndarray, list)) and len(pred_class)>0 else pred_class
         proba_log = f", Proba shape: {pred_proba.shape}" if pred_proba is not None else ""
         logging.info(f"[{target_variable} - {model_name}] Prediction successful: Class={log_class}{proba_log}")
         # Return the full dict and metadata
         return prediction_dict, metadata
    else: # Regression
         pred_value = prediction_dict.get('value')
         log_value = pred_value[0] if isinstance(pred_value, (np.ndarray, list)) and len(pred_value)>0 else pred_value
         logging.info(f"[{target_variable} - {model_name}] Prediction successful: Value={log_value}")
         # Return the full dict and metadata
         return prediction_dict, metadata

# ==========================================
# Main Execution Block
# ==========================================

def main(args):
    logging.info("--- Starting Prediction for Targets: {} ---".format(', '.join(args.targets)))

    input_features_df = None
    input_info = {"mode": args.mode}
    # Assumption: Feature list is consistent. Get from first model/target combo.
    feature_list_source_model = args.model_names[0]
    feature_list_source_target = args.targets[0]
    # Determine model type of the *source* target to load correct metadata for feature list
    source_target_type = 'classification' if feature_list_source_target in settings.AVAILABLE_CLASSIFICATION_TARGETS else 'regression'

    # --- Get Input Features (Done ONCE) ---
    if args.mode == 'auto':
        logging.info(f"Mode: Auto - Fetching live data (using {feature_list_source_model}/{feature_list_source_target} ({source_target_type}) metadata for feature list)...")
        input_info["nifty_ticker"] = args.nifty_ticker
        input_info["vix_ticker"] = args.vix_ticker
        nifty_hist, vix_hist = get_live_data(
            nifty_ticker=args.nifty_ticker, vix_ticker=args.vix_ticker,
            interval=args.interval, period=args.period
        )
        if nifty_hist is None:
            logging.error("Failed to fetch live Nifty data. Aborting.")
            return

        _, temp_metadata = load_model_and_metadata(
            feature_list_source_model, feature_list_source_target, args.week_num, source_target_type
        )
        if temp_metadata is None or constants.FEATURE_COLS not in temp_metadata:
             logging.error(f"Cannot determine feature list from metadata of '{feature_list_source_model}' for target '{feature_list_source_target}'. Aborting auto mode.")
             return
        feature_list = temp_metadata[constants.FEATURE_COLS]
        logging.info(f"Using feature list: {feature_list}")

        input_features_df = prepare_input_features(nifty_hist, vix_hist, feature_list)
        input_info["data_timestamp"] = nifty_hist.index[-1].isoformat() if not nifty_hist.empty else "N/A"
        input_info["fetched_period"] = args.period
        input_info["fetched_interval"] = args.interval
        input_info["features_used"] = ['mon_1h_open', 'mon_1h_high', 'mon_1h_low', 'mon_1h_close', 'vix_mon_open', 'vix_mon_high', 'vix_mon_low', 'vix_mon_close']

    elif args.mode == 'manual':
        logging.info(f"Mode: Manual - Using provided input values (feature list check via {feature_list_source_model}/{feature_list_source_target} ({source_target_type}) metadata)...")
        manual_data = {}
        prefix_map = { # Basic features expected
            'nifty_open': constants.MON_FIRST_HOUR_PREFIX + constants.OPEN,
            'nifty_high': constants.MON_FIRST_HOUR_PREFIX + constants.HIGH,
            'nifty_low': constants.MON_FIRST_HOUR_PREFIX + constants.LOW,
            'nifty_close': constants.MON_FIRST_HOUR_PREFIX + constants.CLOSE,
            'vix_open': constants.VIX_PREFIX_MON + constants.OPEN,
            'vix_high': constants.VIX_PREFIX_MON + constants.HIGH,
            'vix_low': constants.VIX_PREFIX_MON + constants.LOW,
            'vix_close': constants.VIX_PREFIX_MON + constants.CLOSE,
        }
        provided_args = vars(args)
        for arg_name, feature_name in prefix_map.items():
             manual_data[feature_name] = provided_args.get(arg_name, np.nan)

        input_features_df = pd.DataFrame([manual_data])
        input_info["manual_inputs"] = {k:v for k,v in manual_data.items() if pd.notna(v)}

        _, temp_metadata = load_model_and_metadata(
            feature_list_source_model, feature_list_source_target, args.week_num, source_target_type
        )
        if temp_metadata and constants.FEATURE_COLS in temp_metadata:
             model_features = temp_metadata[constants.FEATURE_COLS]
             input_info["features_used_for_check"] = model_features
             for col in model_features:
                 if col not in input_features_df.columns:
                      input_features_df[col] = np.nan
                      logging.warning(f"Feature '{col}' required by model but not in manual args, adding as NaN.")
             try:
                  input_features_df = input_features_df[model_features] # Ensure order and presence
             except KeyError as e:
                  missing_in_df = [f for f in model_features if f not in input_features_df.columns]
                  logging.error(f"Error ensuring columns for manual mode: {e}. Missing required: {missing_in_df}")
                  return
        else:
             logging.warning(f"Could not get full feature list from metadata ('{feature_list_source_model}/{feature_list_source_target}') for manual mode check.")
             input_info["features_used_for_check"] = "Unavailable"

    else:
        logging.error(f"Invalid mode: {args.mode}. Choose 'auto' or 'manual'.")
        return

    if input_features_df is None:
        logging.error("Failed to prepare input features. Aborting prediction.")
        return
    if input_features_df.isnull().values.any():
        logging.warning(f"NaN values present in input features before prediction loop: \n{input_features_df.isnull().sum()[input_features_df.isnull().sum() > 0]}")

    # --- Main Loop: Iterate through Targets ---
    all_target_results = {}
    best_model_metadata_per_target = {}

    for target in args.targets:
        logging.info(f"===== Processing Target: {target} =====")
        current_model_type = 'classification' if target in settings.AVAILABLE_CLASSIFICATION_TARGETS else 'regression'
        logging.info(f"Target type determined as: {current_model_type}")

        target_all_predictions = {} # Store result dicts {'value':...} or {'class':..., 'proba':...}
        target_all_metadata = {}
        target_successful_predictions = {} # Store numeric values (reg) or {'class':..., 'proba':...} (classif)

        # --- Inner Loop: Iterate through Models ---
        logging.info(f"[{target}] Attempting predictions for models: {args.model_names}")
        for model_name in args.model_names:
            prediction_dict, metadata = predict_from_features(
                input_features_df, model_name, target, args.week_num, current_model_type
            )
            target_all_predictions[model_name] = prediction_dict
            target_all_metadata[model_name] = metadata

            if prediction_dict is not None:
                 if current_model_type == 'classification':
                      pred_class = prediction_dict.get('class')
                      pred_proba = prediction_dict.get('proba')
                      if pred_class is not None and pred_proba is not None:
                           cls = pred_class[0] if isinstance(pred_class, (np.ndarray, list)) and len(pred_class)>0 else pred_class
                           prob = pred_proba[0] if isinstance(pred_proba, (np.ndarray, list)) and pred_proba.ndim > 1 and len(pred_proba)>0 else pred_proba
                           target_successful_predictions[model_name] = {'class': cls, 'proba': prob}
                      elif pred_class is not None:
                           cls = pred_class[0] if isinstance(pred_class, (np.ndarray, list)) and len(pred_class)>0 else pred_class
                           target_successful_predictions[model_name] = {'class': cls, 'proba': None}
                           logging.warning(f"[{target} - {model_name}] Storing class prediction but probabilities are missing/invalid.")
                 else: # Regression
                      pred_value = prediction_dict.get('value')
                      if pred_value is not None:
                           val = pred_value[0] if isinstance(pred_value, (np.ndarray, list)) and len(pred_value)>0 else pred_value
                           if isinstance(val, (int, float, np.number)) and not np.isnan(val):
                                target_successful_predictions[model_name] = val

        # --- Calculate Ensemble Prediction (Per Target) ---
        target_ensemble_prediction = None
        if target_successful_predictions:
            if current_model_type == 'classification':
                all_probas = [p['proba'] for p in target_successful_predictions.values() if p.get('proba') is not None]
                if all_probas:
                    try:
                        stacked_probas = np.array(all_probas)
                        if stacked_probas.ndim == 3 and stacked_probas.shape[0] > 0: stacked_probas = stacked_probas.reshape(len(all_probas), -1)
                        elif stacked_probas.ndim != 2: raise ValueError(f"Unexpected probability array dimensions: {stacked_probas.ndim}")

                        avg_proba = np.mean(stacked_probas, axis=0)
                        ensemble_class_index = np.argmax(avg_proba)
                        ensemble_confidence = avg_proba[ensemble_class_index]

                        first_success_model = list(target_successful_predictions.keys())[0]
                        first_meta = target_all_metadata.get(first_success_model)
                        model_classes = first_meta.get('model_classes') if first_meta else None

                        ensemble_class_label = model_classes[ensemble_class_index] if model_classes and len(model_classes) > ensemble_class_index else f"Class_{ensemble_class_index}"

                        target_ensemble_prediction = {'class': ensemble_class_label, 'confidence': ensemble_confidence, 'avg_probabilities': avg_proba.tolist()}
                        logging.info(f"[{target}] Calculated ensemble classification: Class={ensemble_class_label}, Confidence={ensemble_confidence:.4f}")
                    except Exception as ens_e:
                        logging.error(f"[{target}] Error averaging classification probabilities: {ens_e}")
                        logging.debug(f"Probabilities causing error: {all_probas}")
                else: logging.warning(f"[{target}] No valid probabilities found for classification ensemble.")
            else: # Regression
                numeric_values = list(target_successful_predictions.values())
                if numeric_values:
                    # Custom ensemble logic based on target variable
                    if target == 'n_week_high':
                        target_ensemble_prediction = np.max(numeric_values)
                        logging.info(f"[{target}] Calculated ensemble regression (MAX) prediction: {target_ensemble_prediction:.4f} from {len(numeric_values)} models.")
                    elif target == 'n_week_low':
                        target_ensemble_prediction = np.mean(numeric_values)
                        logging.info(f"[{target}] Calculated ensemble regression (MIN) prediction: {target_ensemble_prediction:.4f} from {len(numeric_values)} models.")
                    elif target == 'total_range':
                        target_ensemble_prediction = np.max(numeric_values)
                        logging.info(f"[{target}] Calculated ensemble regression (MAX) prediction: {target_ensemble_prediction:.4f} from {len(numeric_values)} models.")
                    else:
                        # Default to median for other regression targets
                        target_ensemble_prediction = np.median(numeric_values)
                        logging.info(f"[{target}] Calculated ensemble regression (MEDIAN) prediction: {target_ensemble_prediction:.4f} from {len(numeric_values)} models.")
                else: logging.warning(f"[{target}] No successful numerical predictions available for regression ensemble.")
        else: logging.warning(f"[{target}] No successful predictions available to calculate ensemble.")

        # --- Find Best Model Prediction (Per Target) ---
        target_best_model_name = None
        target_best_model_prediction = None # Store value (reg) or dict (classif)
        target_best_metric_value = None
        metric_to_use = args.best_metric
        # Default higher is better for common classification metrics unless it's an error metric
        lower_is_better = metric_to_use.upper() in ['RMSE', 'MAE', 'MSE']

        current_best_val = float('inf') if lower_is_better else -float('inf')

        if target_successful_predictions:
             for model_name, prediction_result in target_successful_predictions.items():
                 metadata = target_all_metadata.get(model_name)
                 if metadata and constants.EVAL_METRICS in metadata:
                     metrics_dict = metadata.get(constants.EVAL_METRICS, {})
                     metric_val = metrics_dict.get(metric_to_use.upper()) or metrics_dict.get(metric_to_use.lower())

                     if metric_val is not None:
                         logging.debug(f"[{target} - {model_name}]: Found metric {metric_to_use} = {metric_val}")
                         if lower_is_better:
                             if metric_val < current_best_val:
                                 current_best_val = metric_val
                                 target_best_model_name = model_name
                                 target_best_model_prediction = prediction_result
                         else: # Higher is better
                             if metric_val > current_best_val:
                                 current_best_val = metric_val
                                 target_best_model_name = model_name
                                 target_best_model_prediction = prediction_result
                     else: logging.warning(f"[{target} - {model_name}] Metric '{metric_to_use}' not found in metadata. Available: {list(metrics_dict.keys())}")
                 else: logging.warning(f"[{target} - {model_name}] Metadata or evaluation metrics missing.")

             if target_best_model_name:
                 target_best_metric_value = current_best_val
                 best_model_metadata_per_target[target] = target_all_metadata.get(target_best_model_name)
                 log_pred_val = target_best_model_prediction.get('class', target_best_model_prediction) if isinstance(target_best_model_prediction, dict) else target_best_model_prediction
                 logging.info(f"[{target}] Best model based on {metric_to_use} ({'lower' if lower_is_better else 'higher'} is better): '{target_best_model_name}' (Metric: {target_best_metric_value:.4f}, Prediction: {log_pred_val})")
             else: logging.warning(f"[{target}] Could not determine the best model based on metric '{metric_to_use}'.")
        else: logging.warning(f"[{target}] No successful predictions available to determine the best model.")

        # --- Store results for this target ---
        all_target_results[target] = {
            "prediction_type": current_model_type,
            "individual_predictions": target_all_predictions, # Includes None for failures
            "ensemble_prediction": target_ensemble_prediction,
            # "best_model_details": {
            #     "model_name": target_best_model_name,
            #     "prediction": target_best_model_prediction,
            #     "metric_used": metric_to_use,
            #     "metric_value": target_best_metric_value
            # }
        }
        logging.info(f"===== Finished Processing Target: {target} =====")

    # --- Format Final Output ---
    final_output_dict = {
        "target_predictions": all_target_results,
        # "best_models_metadata": best_model_metadata_per_target
    }
    final_output_str = format_prediction_output( # Use the local version
        final_output_dict, {}, input_info
    )
    print(final_output_str)

# ==========================================
# Argparse Setup and Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using trained models for multiple targets, optionally ensembling.")

    # --- Model/Target Specification ---
    parser.add_argument('--model_names', type=str, required=True, nargs='+',
                        help='Name(s) of the trained model file(s) (e.g., XGBRegressor RandomForestClassifier).')
    parser.add_argument('--targets', type=str, required=True, nargs='+',
                        help='Target variable(s) (regression and/or classification) to predict (e.g., total_range price_direction).')
    parser.add_argument('--week_num', type=int, required=True,
                        help='Expiry week number the models were trained for (1-4).')
    parser.add_argument('--best_metric', type=str, default='RMSE',
                        help='Metric from saved metadata to determine the "best" model per target (e.g., RMSE, Accuracy, F1). Default: RMSE.')

    # --- Input Mode ---
    parser.add_argument('--mode', type=str, required=True, choices=['auto', 'manual'],
                        help='Input mode: "auto" (fetch from yfinance) or "manual" (provide values).')

    # --- Auto Mode Arguments ---
    parser.add_argument('--nifty_ticker', type=str, default='^NSEI', help='Nifty ticker symbol for yfinance (auto mode).')
    parser.add_argument('--vix_ticker', type=str, default='^INDIAVIX', help='VIX ticker symbol for yfinance (auto mode).')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval for yfinance (e.g., 1h, 1d) (auto mode).')
    parser.add_argument('--period', type=str, default='7d', help='Data period for yfinance (e.g., 7d, 1mo) (auto mode).') # Default 7d

    # --- Manual Mode Arguments ---
    parser.add_argument('--nifty_open', type=float, help='Monday/Tuesday first hour Nifty Open (manual mode).')
    parser.add_argument('--nifty_high', type=float, help='Monday/Tuesday first hour Nifty High (manual mode).')
    parser.add_argument('--nifty_low', type=float, help='Monday/Tuesday first hour Nifty Low (manual mode).')
    parser.add_argument('--nifty_close', type=float, help='Monday/Tuesday first hour Nifty Close (manual mode).')
    parser.add_argument('--vix_open', type=float, help='Monday/Tuesday VIX Open (manual mode).')
    parser.add_argument('--vix_high', type=float, help='Monday/Tuesday VIX High (manual mode).')
    parser.add_argument('--vix_low', type=float, help='Monday/Tuesday VIX Low (manual mode).')
    parser.add_argument('--vix_close', type=float, help='Monday/Tuesday VIX Close (manual mode).')

    args = parser.parse_args()
    main(args)