# backtest.py
import argparse
import logging
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import pytz

# Assume these modules exist in your project structure
# Make sure constants includes VALID_TARGETS
from config import constants, settings
from utils.preprocessing import load_data, find_trading_day
from utils.predictions import load_model_and_metadata, make_prediction, prepare_input_features
from utils.evaluation import calculate_regression_metrics # Use if evaluating performance vs actual

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(args):
    """Performs historical testing based on input dates."""
    logging.info("--- Starting Historical Backtest ---")

    # --- 1. Load Input Dates ---
    try:
        test_dates_df = pd.read_csv(args.input_csv)
        # Ensure date column is parsed correctly as date objects (without time)
        test_dates_df[args.date_col] = pd.to_datetime(test_dates_df[args.date_col]).dt.date
        logging.info(f"Loaded {len(test_dates_df)} dates for backtesting.")
    except FileNotFoundError:
        logging.error(f"Error: Input CSV file not found at '{args.input_csv}'")
        return
    except KeyError:
        logging.error(f"Error: Date column '{args.date_col}' not found in '{args.input_csv}'")
        return
    except Exception as e:
        logging.error(f"Error loading backtest input CSV '{args.input_csv}': {e}")
        return

    # --- 2. Load Historical Nifty & VIX Data (Load ONCE) ---
    logging.info(f"Loading Nifty data from: {args.nifty_csv}")
    nifty_history = load_data(args.nifty_csv)
    if nifty_history is None:
        logging.error("Failed to load Nifty data. Exiting.")
        return
    # Ensure Nifty index is DatetimeIndex
    if not isinstance(nifty_history.index, pd.DatetimeIndex):
        logging.error("Nifty data index is not a DatetimeIndex. Please preprocess data.")
        return

    logging.info(f"Loading VIX data from: {args.vix_csv}")
    vix_history = load_data(args.vix_csv)
    if vix_history is not None and not isinstance(vix_history.index, pd.DatetimeIndex):
         logging.warning("VIX data index is not a DatetimeIndex. Will proceed but features might fail.")
    # VIX failure might be acceptable if models don't use it, handle inside loop

    # --- 3. Initialize Results Storage ---
    results_list = []
    consecutive_failures = 0 # Track consecutive prediction failures

    # --- 4. Loop Through Test Dates ---
    for index, row in test_dates_df.iterrows():
        test_date = row[args.date_col]
        # Ensure test_date is a date object, not timestamp
        if isinstance(test_date, pd.Timestamp):
             test_date = test_date.date()

        logging.info(f"--- Backtesting Date: {test_date} ---")
        # Include original row data from input CSV in results
        test_date_result = {'test_date': test_date, **row.to_dict()}

        # --- 5. Prepare Features for test_date ---
        # Use data ONLY UP TO the end of the test_date.
        try:
             # Define the end boundary for slicing historical data (inclusive of test_date)
             end_filter_dt = pd.Timestamp(test_date, tz=constants.TIMEZONE).normalize() + timedelta(days=1) - timedelta.resolution

             # Slice data up to the end of the test date
             nifty_slice = nifty_history.loc[:end_filter_dt]
             vix_slice = vix_history.loc[:end_filter_dt] if vix_history is not None else None

             # --- Get feature list ---
             # Load metadata just once per date, assuming features are consistent across models for the same target type
             # We still need to load it here to know *which* features are needed.
             # Use the first model/target combo just to fetch the metadata for features.
             # A more robust solution might involve a separate feature configuration.
             first_model_name = args.model_names[0]
             first_target_name = args.targets[0]
             _, temp_metadata = load_model_and_metadata(first_model_name, first_target_name, args.week_num)

             if temp_metadata is None or constants.FEATURE_COLS not in temp_metadata:
                  logging.error(f"Cannot get feature list from metadata for {first_model_name}/{first_target_name}. Skipping date {test_date}.")
                  test_date_result['status'] = 'Metadata/Feature Error'
                  results_list.append(test_date_result)
                  continue
             feature_list = temp_metadata[constants.FEATURE_COLS]
             logging.debug(f"Using feature list for {test_date}: {feature_list}")

             # --- Prepare features based *only* on data up to the test date ---
             # prepare_input_features should find the latest suitable day within the slice
             input_features_df = prepare_input_features(nifty_slice, vix_slice, feature_list)

             if input_features_df is None or input_features_df.empty:
                  logging.warning(f"Could not prepare input features for {test_date} using features: {feature_list}. Skipping.")
                  test_date_result['status'] = 'Feature Prep Failed'
                  results_list.append(test_date_result)
                  continue
             # Log the prepared features for debugging if needed
             # logging.debug(f"Prepared features for {test_date}:\n{input_features_df.iloc[-1:]}") # Show last row
             test_date_result['status'] = 'Features Prepared'

        except Exception as feat_e:
             logging.exception(f"Error preparing features for {test_date}: {feat_e}") # Use logging.exception for traceback
             test_date_result['status'] = 'Feature Prep Error'
             results_list.append(test_date_result)
             continue


        # --- 6. Generate Predictions (Loop through Targets & Models) ---
        target_predictions = {}
        prediction_successful_flag = False # Flag if any valid numeric prediction was made

        for target in args.targets: # Iterate through targets requested by user
            target_preds_for_date = {}
            target_successful_numeric = {} # Store only valid numeric predictions for ensemble

            for model_name in args.model_names: # Iterate through models requested by user
                pred_val = None # Reset for each model
                try:
                    # Load model and metadata for the specific model/target combo
                    model_object, metadata = load_model_and_metadata(model_name, target, args.week_num)

                    if model_object is not None:
                         # Make prediction using the *single row* of prepared features
                         # Ensure the features used match what this specific model was trained on,
                         # This assumes prepare_input_features generates features suitable for all models,
                         # or that metadata check above is sufficient.
                         pred_val = make_prediction(model_object, input_features_df[feature_list]) # Ensure correct features used

                         if pred_val is not None:
                              # Extract single value if prediction returns array/list
                              pred_val = pred_val[0] if isinstance(pred_val, (np.ndarray, list)) and len(pred_val) > 0 else pred_val
                              target_preds_for_date[model_name] = pred_val # Store raw prediction (could be NaN)

                              # Check if prediction is a valid number for ensemble/success flag
                              if isinstance(pred_val, (int, float, np.number)) and not np.isnan(pred_val):
                                   target_successful_numeric[model_name] = pred_val
                                   prediction_successful_flag = True # Mark success if at least one model gives valid output
                                   logging.debug(f"Prediction successful for {model_name}/{target} on {test_date}: {pred_val}")
                              else:
                                  logging.warning(f"Prediction for {model_name}/{target} on {test_date} resulted in None or NaN.")

                         else:
                              logging.warning(f"Prediction function returned None for {model_name}/{target} on {test_date}")
                              target_preds_for_date[model_name] = None # Prediction function failed
                    else:
                         logging.warning(f"Model loading failed for {model_name}/{target} on {test_date}")
                         target_preds_for_date[model_name] = None # Model loading failed

                except Exception as pred_e:
                    logging.exception(f"Error during prediction for {model_name}/{target} on {test_date}: {pred_e}")
                    target_preds_for_date[model_name] = None # Mark error

            # Calculate ensemble (e.g., mean) for the target using only successful numeric predictions
            ensemble_pred = np.nanmean(list(target_successful_numeric.values())) if target_successful_numeric else np.nan # Use nanmean, fallback to nan
            target_predictions[target] = {
                'individual': target_preds_for_date,
                'ensemble': ensemble_pred
            }
            logging.debug(f"Ensemble prediction for {target} on {test_date}: {ensemble_pred}")

        test_date_result['predictions'] = target_predictions # Store nested dict

        # --- 7. Load Actual Future Data & Calculate Actuals ---
        try:
             # Determine the target date range for actuals
             future_start_target_date = test_date + timedelta(days=1)
             future_end_target_date = test_date + timedelta(weeks=args.week_num)

             # Find the actual trading days corresponding to the target dates
             available_nifty_dates = pd.Series(nifty_history.index.date).unique()
             available_nifty_dates.sort()

             actual_start_trading_day = find_trading_day(future_start_target_date, available_nifty_dates, 'forward')
             actual_end_trading_day = find_trading_day(future_end_target_date, available_nifty_dates, 'backward')

             # --- Validate the found actual trading days ---
             if actual_start_trading_day is None or actual_end_trading_day is None:
                 logging.warning(f"Could not find actual start/end trading day for {test_date} -> w{args.week_num}. Skipping actuals.")
                 test_date_result['status'] = 'Actuals Trading Days Not Found'
                 actuals_calculated = False
             elif actual_end_trading_day < actual_start_trading_day:
                  logging.warning(f"Actual end trading day ({actual_end_trading_day}) is before start day ({actual_start_trading_day}) for {test_date}. Skipping actuals.")
                  test_date_result['status'] = 'Actuals Date Range Invalid'
                  actuals_calculated = False
             else:
                 # --- Define precise timestamps for filtering Nifty data ---
                 start_dt_filter = pd.Timestamp(actual_start_trading_day, tz=constants.TIMEZONE).normalize()
                 end_dt_filter = pd.Timestamp(actual_end_trading_day, tz=constants.TIMEZONE).normalize() + timedelta(days=1) - timedelta.resolution

                 # --- Extract the Nifty data for the actual period ---
                 future_period_data = nifty_history.loc[start_dt_filter:end_dt_filter]

                 if future_period_data.empty:
                      logging.warning(f"No Nifty data found in actual future period ({actual_start_trading_day} to {actual_end_trading_day}) for {test_date}.")
                      test_date_result['status'] = 'Actuals Data Empty'
                      actuals_calculated = False
                 else:
                      # --- Calculate Actual High, Low, Range ---
                      actual_high = future_period_data[constants.HIGH].max()
                      actual_low = future_period_data[constants.LOW].min()
                      actual_total_range = actual_high - actual_low

                      # --- Calculate Actual Close Difference ---
                      start_day_data = nifty_history.loc[nifty_history.index.date == test_date]
                      start_day_close = start_day_data[constants.CLOSE].iloc[-1] if not start_day_data.empty else np.nan

                      end_day_data = nifty_history.loc[nifty_history.index.date == actual_end_trading_day]
                      end_day_close = end_day_data[constants.CLOSE].iloc[-1] if not end_day_data.empty else np.nan # This is the value we need

                      actual_close_diff = end_day_close - start_day_close if pd.notna(start_day_close) and pd.notna(end_day_close) else np.nan

                      # --- Store the calculated actuals ---
                      test_date_result['actual_start_date'] = actual_start_trading_day
                      test_date_result['actual_end_date'] = actual_end_trading_day
                      test_date_result['actual_high'] = actual_high
                      test_date_result['actual_low'] = actual_low
                      test_date_result['actual_total_range'] = actual_total_range
                      test_date_result['actual_close_diff'] = actual_close_diff
                      test_date_result['actual_close'] = end_day_close # Store actual close
                      test_date_result['status'] = 'Completed' # Mark as successfully processed
                      actuals_calculated = True
                      logging.debug(f"Actuals calculated for {test_date}: High={actual_high}, Low={actual_low}, Close={end_day_close}")


             # --- Ensure actual columns exist even if calculation failed ---
             if not actuals_calculated:
                 test_date_result.setdefault('actual_start_date', actual_start_trading_day if 'actual_start_trading_day' in locals() else np.nan)
                 test_date_result.setdefault('actual_end_date', actual_end_trading_day if 'actual_end_trading_day' in locals() else np.nan)
                 test_date_result.setdefault('actual_high', np.nan)
                 test_date_result.setdefault('actual_low', np.nan)
                 test_date_result.setdefault('actual_total_range', np.nan)
                 test_date_result.setdefault('actual_close_diff', np.nan)
                 test_date_result.setdefault('actual_close', np.nan)


             # --- 8. Evaluate Predictions vs Actuals (Optional - Basic Example) ---
             eval_metrics = {}
             if actuals_calculated: # Only evaluate if actuals are available
                 # Example evaluation using 'n_week_high' target if present
                 eval_target = 'n_week_high' # Choose a target to evaluate
                 if eval_target in target_predictions and 'ensemble' in target_predictions[eval_target]:
                     pred_val = target_predictions[eval_target]['ensemble']
                     actual_val = test_date_result['actual_high'] # Compare against actual high
                     if pd.notna(pred_val) and pd.notna(actual_val):
                         eval_metrics[f'{eval_target}_ensemble_error'] = pred_val - actual_val
                         if actual_val != 0:
                              eval_metrics[f'{eval_target}_ensemble_pct_error'] = (pred_val - actual_val) / actual_val
                         else:
                              eval_metrics[f'{eval_target}_ensemble_pct_error'] = np.nan
                     else:
                         eval_metrics[f'{eval_target}_ensemble_error'] = np.nan
                         eval_metrics[f'{eval_target}_ensemble_pct_error'] = np.nan
                 else:
                    logging.debug(f"Ensemble prediction for '{eval_target}' not available for evaluation on {test_date}")

             test_date_result['evaluation'] = eval_metrics # Store evaluation metrics


             # --- 9. Consecutive Failure Check ---
             # Define "failure" based on prediction success flag
             is_failure = not prediction_successful_flag # Failure if NO model produced a valid numeric prediction
             if is_failure:
                  consecutive_failures += 1
                  logging.warning(f"Prediction failed (no valid numeric output) for date {test_date}. Consecutive failures: {consecutive_failures}")
             else:
                  # Reset only if predictions were successful AND actuals were calculated (or actuals step didn't error out)
                  # This prevents resetting failure count if prediction worked but actuals failed later.
                  if test_date_result.get('status') not in ['Actuals/Eval Error', 'Actuals Data Empty', 'Actuals Date Range Invalid', 'Actuals Trading Days Not Found']:
                       consecutive_failures = 0

             test_date_result['consecutive_failures'] = consecutive_failures
             test_date_result['failure_alert'] = consecutive_failures >= settings.BACKTEST_CONSECUTIVE_FAILURE_THRESHOLD


        except Exception as actual_e:
            logging.exception(f"Error processing actuals or evaluation for {test_date}: {actual_e}")
            test_date_result['status'] = 'Actuals/Eval Error'
            # Ensure actual columns exist even if error occurred, set to NaN
            test_date_result.setdefault('actual_start_date', np.nan)
            test_date_result.setdefault('actual_end_date', np.nan)
            test_date_result.setdefault('actual_high', np.nan)
            test_date_result.setdefault('actual_low', np.nan)
            test_date_result.setdefault('actual_total_range', np.nan)
            test_date_result.setdefault('actual_close_diff', np.nan)
            test_date_result.setdefault('actual_close', np.nan) # Ensure actual_close is present
            test_date_result.setdefault('evaluation', {})
            test_date_result.setdefault('consecutive_failures', consecutive_failures) # Report current count
            test_date_result.setdefault('failure_alert', consecutive_failures >= settings.BACKTEST_CONsecutive_FAILURE_THRESHOLD)


        # --- 10. Append result for the date ---
        results_list.append(test_date_result)
        logging.info(f"Finished processing {test_date}. Status: {test_date_result.get('status', 'Unknown')}")


    # --- 11. Process and Save Final Results ---
    if not results_list:
        logging.warning("No results were generated during the backtest.")
        return

    logging.info("Processing final results...")
    try:
        # Use json_normalize to flatten nested dictionaries
        final_results_df = pd.json_normalize(results_list, sep='_')

        # Ensure date columns are in 'YYYY-MM-DD' format, handling potential NaT/None
        date_cols_to_format = ['test_date', 'actual_start_date', 'actual_end_date']
        for col in date_cols_to_format:
            if col in final_results_df.columns:
                 # Convert to datetime, coercing errors, then format
                 final_results_df[col] = pd.to_datetime(final_results_df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                 # Replace NaT strings resulting from formatting NaT dates
                 final_results_df[col] = final_results_df[col].replace('NaT', np.nan)


        # Attempt to infer better dtypes for numeric columns that might be object
        final_results_df = final_results_df.infer_objects()

        final_results_df.to_csv(args.output_csv, index=False)
        logging.info(f"Backtest results saved to {args.output_csv}")

    except Exception as save_e:
        logging.exception(f"Failed to process or save backtest results: {save_e}")
        # Fallback: try saving the raw list of dicts if normalization fails
        try:
            fallback_path = args.output_csv.replace(".csv", "_raw_dicts.csv")
            pd.DataFrame(results_list).to_csv(fallback_path, index=False)
            logging.info(f"Saved raw results (list of dicts) to {fallback_path}")
        except Exception as raw_save_e:
            logging.error(f"Could not even save raw results: {raw_save_e}")


    logging.info("--- Historical Backtest Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run historical backtest for prediction models.")

    # --- File/Data Arguments ---
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file containing dates column for backtesting.')
    parser.add_argument('--date_col', type=str, default='Date', # Changed default to 'Date' based on user command
                        help='Name of the date column in the input CSV.')
    parser.add_argument('--nifty_csv', type=str, default=constants.NIFTY_RAW_FILE,
                        help=f'Path to the raw Nifty data CSV file (required). Default: {constants.NIFTY_RAW_FILE}')
    parser.add_argument('--vix_csv', type=str, default=constants.VIX_RAW_FILE,
                        help=f'Path to the raw VIX data CSV file (optional, depends on model features). Default: {constants.VIX_RAW_FILE}')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save the detailed backtest results CSV.')

    # --- Model/Prediction Arguments ---
    parser.add_argument('--model_names', type=str, required=True, nargs='+',
                        help='Name(s) of the trained model file(s) (without path/extension, e.g., LGBMRegressor RandomForestRegressor).')

    # --- CORRECTED TARGETS ARGUMENT ---
    # Make sure constants.VALID_TARGETS exists and is a list/tuple in your config/constants.py
    if hasattr(constants, 'VALID_TARGETS') and isinstance(constants.VALID_TARGETS, (list, tuple)):
        parser.add_argument('--targets', type=str, required=True, nargs='+',
                            choices=constants.VALID_TARGETS,
                            help=f'Target variable(s) the models predict (choose from {constants.VALID_TARGETS}).')
    else:
        logging.warning("constants.VALID_TARGETS not found or not a list/tuple. Defining --targets without choices.")
        parser.add_argument('--targets', type=str, required=True, nargs='+',
                            help='Target variable(s) the models predict (e.g., n_week_high n_week_low). Validation skipped.')


    parser.add_argument('--week_num', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Expiry week number (1-4) the models were trained for.')
    # Feature set selection might be implicitly handled by model metadata or could be an explicit argument
    # parser.add_argument('--feature_set', type=str, default=settings.DEFAULT_FEATURE_SET, help='Feature set identifier.')

    args = parser.parse_args()

    # Basic Validation
    valid_run = True
    if not os.path.exists(args.input_csv):
         logging.error(f"Input dates file not found: {args.input_csv}")
         valid_run = False
    if not os.path.exists(args.nifty_csv):
        logging.error(f"Nifty data file not found: {args.nifty_csv}")
        valid_run = False
    # Optional: Check for VIX file existence if it's always needed
    # if not os.path.exists(args.vix_csv) and VIX_IS_MANDATORY:
    #      logging.error(f"VIX data file not found: {args.vix_csv}")
    #      valid_run = False

    if valid_run:
        run_backtest(args)
    else:
        logging.error("Exiting due to missing input files.")