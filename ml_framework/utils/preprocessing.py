# utils/preprocessing.py
import pandas as pd
import numpy as np
import os
import logging
from datetime import time, timedelta, date
import pytz
from statsmodels.tsa.stattools import adfuller

from config.constants import (
    DATETIME, OPEN, HIGH, LOW, CLOSE, # Removed VOLUME, VOLUME_MA
    NIFTY_RAW_FILE, VIX_RAW_FILE,
    PROCESSED_DATA_DIR, TIMEZONE, DATETIME_FORMAT, MON_FIRST_HOUR_PREFIX, EXP_PREFIX,
    WEEK_NUM, MON_DATE, EXP_DATE, RANGE_HIGH, RANGE_LOW, TOTAL_RANGE,
    MON_CLOSE_MINUS_EXP_CLOSE, N_WEEK_HIGH, N_WEEK_LOW,
    N_WEEK_HIGH_MINUS_MON_CLOSE, N_WEEK_LOW_MINUS_MON_CLOSE,
    N_WEEK_HIGH_MINUS_THURSDAY_CLOSE, N_WEEK_LOW_MINUS_THURSDAY_CLOSE,
    VIX_PREFIX_MON, VIX_PREFIX_EXP, CLASSIFICATION_TARGET_RANGE_BINS, CALCULATE_MOMENTUM, ROLLING_WINDOWS, CALCULATE_VOLATILITY, CALCULATE_ROLLING_STATS, ADFULLER_SIGNIFICANCE_LEVEL, 
    MON_FIRST_HOUR_PREFIX, VIX_PREFIX_EXP, VIX_PREFIX_MON,
)
from config.settings import FIRST_HOUR_START, FIRST_HOUR_END, LAST_HOUR_START, LAST_HOUR_END, AVAILABLE_CLASSIFICATION_TARGETS, MON_FIRST_HOUR_PREFIX, VIX_PREFIX_EXP, VIX_PREFIX_MON

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- load_data function updated ---
def load_data(file_path, date_col=DATETIME, date_format=DATETIME_FORMAT, tz=TIMEZONE):
    """Loads CSV data, parses datetime, and sets timezone."""
    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
        except ValueError:
             logging.warning(f"Could not parse all dates with format {date_format}. Trying generic parsing.")
             df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        df = df.dropna(subset=[date_col]) # Drop rows where date couldn't be parsed

        if df[date_col].dt.tz is None:
            logging.info(f"Localizing timezone-naive datetime column to {tz}")
            df[date_col] = df[date_col].dt.tz_localize(tz)
        else:
            logging.info(f"Converting datetime column timezone to {tz}")
            df[date_col] = df[date_col].dt.tz_convert(tz)

        df = df.sort_values(by=date_col)
        df = df.set_index(date_col)
        logging.info(f"Data loaded successfully with {len(df)} rows.")
        # Ensure essential columns are numeric (Removed VOLUME, VOLUME_MA)
        for col in [OPEN, HIGH, LOW, CLOSE]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Removed VOLUME/VOLUME_MA checks
        df = df.dropna(subset=[OPEN, HIGH, LOW, CLOSE]) # Drop if core values are missing
        return df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

# --- find_expiry_thursday function ---
def find_expiry_thursday(monday_date, week_offset):
    """Finds the Nth Thursday on or after the Monday."""
    days_until_thursday = (3 - monday_date.weekday() + 7) % 7
    first_thursday = monday_date + timedelta(days=days_until_thursday)
    expiry_thursday = first_thursday + timedelta(weeks=week_offset - 1)
    return expiry_thursday

# --- ADD THIS FUNCTION ---
def find_trading_day(target_date, available_dates, direction='forward'):
    """
    Finds the nearest available trading day in the specified direction.

    Args:
        target_date (datetime.date): The date to start searching from.
        available_dates (pd.Series or np.array): Sorted unique dates available in the data.
        direction (str): 'forward' or 'backward'.

    Returns:
        datetime.date or None: The nearest trading day found, or None if none found matching criteria.
    """
    if not isinstance(target_date, date):
         logging.warning(f"target_date is not a date object: {target_date}. Attempting conversion.")
         try:
             target_date = pd.to_datetime(target_date).date()
         except Exception:
              logging.error(f"Could not convert target_date {target_date} to date.")
              return None

    # Ensure available_dates is suitable for comparison (needs to contain date objects)
    if not isinstance(available_dates, (pd.Series, np.ndarray)):
         logging.error("available_dates must be a pandas Series or numpy array.")
         return None
    # Convert available_dates elements to date objects if they are not already
    # This can be slow if done repeatedly, better to ensure it's done once outside the loop
    # For robustness here, we add a check/conversion (assuming elements are datetime-like)
    if len(available_dates) > 0 and not isinstance(available_dates[0], date):
         try:
              # Optimization: convert only if needed
              if not all(isinstance(d, date) for d in available_dates):
                   available_dates = pd.to_datetime(available_dates).date
         except Exception as e:
              logging.error(f"Could not convert elements of available_dates to date objects: {e}")
              return None


    try:
        if direction == 'forward':
            # Find the first date in available_dates >= target_date
            possible_dates = available_dates[available_dates >= target_date]
            # Use min() which works on Series/arrays of dates
            return possible_dates.min() if len(possible_dates) > 0 else None
        elif direction == 'backward':
            # Find the last date in available_dates <= target_date
            possible_dates = available_dates[available_dates <= target_date]
            # Use max() which works on Series/arrays of dates
            return possible_dates.max() if len(possible_dates) > 0 else None
        else:
            logging.error(f"Invalid direction specified: {direction}")
            return None
    except TypeError as te:
        logging.error(f"TypeError finding trading day for {target_date} ({direction}): {te}. Check date types.")
        return None
    except Exception as e:
         logging.error(f"Error finding trading day for {target_date} ({direction}): {e}")
         return None

# --- get_first_hour_data remains the same ---
def get_first_hour_data(df, target_date, start_time_str, end_time_str):
    """Helper to get first hour data for a specific date."""
    start_time = time.fromisoformat(start_time_str)
    end_time = time.fromisoformat(end_time_str)
    return df[
        (df['date'] == target_date) &
        (df['time'] >= start_time) &
        (df['time'] < end_time) # End time is exclusive
    ]

# --- aggregate_data function updated ---
def aggregate_data(data_df, prefix="", open_col=OPEN, high_col=HIGH, low_col=LOW, close_col=CLOSE):
    """Helper to aggregate OHLC data."""
    if data_df is None or data_df.empty:
        # Return a Series with NaNs if input is empty
        return pd.Series(index=[f"{prefix}{col}" for col in [open_col, high_col, low_col, close_col]], dtype=float)

    agg_dict = {
        open_col: lambda x: x.iloc[0] if not x.empty else np.nan,
        high_col: 'max',
        low_col: 'min',
        close_col: lambda x: x.iloc[-1] if not x.empty else np.nan,
    }
    # Removed Volume/VolMA aggregations

    agg_result = data_df.agg(agg_dict)

    return agg_result.add_prefix(prefix)


def process_weekly_expiry_data(nifty_df, vix_df, output_dir=PROCESSED_DATA_DIR):
    """
    Processes Nifty and VIX data to create weekly expiry datasets (OHLC only),
    handling missing Monday/Thursday by shifting to Tue/Wed.
    """
    if nifty_df is None or vix_df is None:
        logging.error("Input Nifty or VIX dataframe is None. Aborting processing.")
        return

    logging.info("Starting weekly expiry data processing...")

    nifty_df['date'] = nifty_df.index.date
    nifty_df['time'] = nifty_df.index.time
    nifty_df['weekday'] = nifty_df.index.weekday # Monday = 0

    mondays_in_data = sorted(nifty_df[nifty_df['weekday'] == 0]['date'].unique())
    logging.info(f"Found {len(mondays_in_data)} unique Mondays in Nifty data.")

    vix_df['date'] = vix_df.index.date
    vix_df['time'] = vix_df.index.time # <--- ADD THIS LINE

    processed_data_all_weeks = {1: [], 2: [], 3: [], 4: []}

    for mon_date_obj in mondays_in_data:
        logging.debug(f"Processing Target Monday: {mon_date_obj}")

        # --- 1. Get Monday/Tuesday First Hour Data ---
        actual_start_date_obj = None
        mon_first_hour_data = get_first_hour_data(nifty_df, mon_date_obj, FIRST_HOUR_START, FIRST_HOUR_END)

        if mon_first_hour_data.empty:
            logging.warning(f"No first hour data found for Monday {mon_date_obj}. Trying Tuesday.")
            tue_date_obj = mon_date_obj + timedelta(days=1)
            mon_first_hour_data = get_first_hour_data(nifty_df, tue_date_obj, FIRST_HOUR_START, FIRST_HOUR_END)

            if mon_first_hour_data.empty:
                logging.warning(f"No first hour data found for fallback Tuesday {tue_date_obj}. Skipping this week start.")
                continue
            else:
                logging.info(f"Using Tuesday {tue_date_obj} data as start day.")
                actual_start_date_obj = tue_date_obj
        else:
            actual_start_date_obj = mon_date_obj # Monday data found

        # Aggregate Monday/Tuesday first hour OHLC
        mon_1h_agg = aggregate_data(mon_first_hour_data, MON_FIRST_HOUR_PREFIX)

        # Get VIX data for the *actual* start date's first hour
        vix_mon_data = get_first_hour_data(vix_df, actual_start_date_obj, FIRST_HOUR_START, FIRST_HOUR_END)
        vix_mon_agg = aggregate_data(vix_mon_data, VIX_PREFIX_MON) # Use helper
        if vix_mon_data.empty:
             logging.warning(f"No first hour VIX data found for start date {actual_start_date_obj}. Using NaNs.")


        # Iterate through week offsets (1 to 4)
        for week_offset in range(1, 5):
            target_exp_date_obj = find_expiry_thursday(mon_date_obj, week_offset) # Calculate based on original Monday
            logging.debug(f"  Week {week_offset}: Target expiry Thursday: {target_exp_date_obj}")

            # --- 2. Get Expiry Day (Thursday/Wednesday) Data ---
            actual_exp_date_obj = None
            exp_day_data = nifty_df[nifty_df['date'] == target_exp_date_obj]

            if exp_day_data.empty:
                logging.warning(f"No Nifty data found for target expiry Thursday {target_exp_date_obj}. Trying Wednesday.")
                wed_date_obj = target_exp_date_obj - timedelta(days=1)
                exp_day_data = nifty_df[nifty_df['date'] == wed_date_obj] # Reassign

                if exp_day_data.empty:
                    logging.warning(f"No Nifty data found for fallback Wednesday {wed_date_obj}. Using NaNs for expiry metrics.")
                    actual_exp_date_obj = target_exp_date_obj # Keep target for range calc consistency
                    # Create NaN series for expiry OHLC (Removed Volume/VolMA)
                    exp_ohlcv = pd.Series(index=[f"{EXP_PREFIX}{col}" for col in [OPEN, HIGH, LOW, CLOSE]], dtype=float)
                    exp_last_hour_close = np.nan
                    vix_exp_agg = pd.Series(index=[f"{VIX_PREFIX_EXP}{col}" for col in [OPEN, HIGH, LOW, CLOSE]], dtype=float) # VIX also NaN

                else:
                    logging.info(f"Using data from Wednesday {wed_date_obj} as expiry day.")
                    actual_exp_date_obj = wed_date_obj
                    # Aggregate Wednesday OHLC data
                    exp_ohlcv = aggregate_data(exp_day_data, EXP_PREFIX)
                    # Get Wednesday's last hour close
                    exp_last_hour_data = exp_day_data[
                        (exp_day_data.index.time >= time.fromisoformat(LAST_HOUR_START)) &
                        (exp_day_data.index.time <= time.fromisoformat(LAST_HOUR_END))
                    ]
                    exp_last_hour_close = exp_last_hour_data[CLOSE].iloc[-1] if not exp_last_hour_data.empty else exp_ohlcv.get(f"{EXP_PREFIX}{CLOSE}", np.nan)

                    # Get VIX data for *actual* expiry date (Wednesday) last hour
                    vix_exp_data = vix_df[
                        (vix_df['date'] == actual_exp_date_obj) &
                        (vix_df.index.time >= time.fromisoformat(LAST_HOUR_START)) &
                        (vix_df.index.time <= time.fromisoformat(LAST_HOUR_END))
                    ]
                    vix_exp_agg = aggregate_data(vix_exp_data, VIX_PREFIX_EXP) # Use helper
                    if vix_exp_data.empty:
                         logging.warning(f"No last hour VIX data found for actual expiry date {actual_exp_date_obj}. Using NaNs.")

            else:
                # Thursday data was found
                actual_exp_date_obj = target_exp_date_obj
                logging.debug(f"Found Nifty data for target expiry Thursday {actual_exp_date_obj}.")
                # Aggregate Thursday OHLC data
                exp_ohlcv = aggregate_data(exp_day_data, EXP_PREFIX)
                # Get Thursday's last hour close
                exp_last_hour_data = exp_day_data[
                    (exp_day_data.index.time >= time.fromisoformat(LAST_HOUR_START)) &
                    (exp_day_data.index.time <= time.fromisoformat(LAST_HOUR_END))
                ]
                exp_last_hour_close = exp_last_hour_data[CLOSE].iloc[-1] if not exp_last_hour_data.empty else exp_ohlcv.get(f"{EXP_PREFIX}{CLOSE}", np.nan)

                # Get VIX data for *actual* expiry date (Thursday) last hour
                vix_exp_data = vix_df[
                    (vix_df['date'] == actual_exp_date_obj) &
                    (vix_df.index.time >= time.fromisoformat(LAST_HOUR_START)) &
                    (vix_df.index.time <= time.fromisoformat(LAST_HOUR_END))
                ]
                vix_exp_agg = aggregate_data(vix_exp_data, VIX_PREFIX_EXP) # Use helper
                if vix_exp_data.empty:
                     logging.warning(f"No last hour VIX data found for actual expiry date {actual_exp_date_obj}. Using NaNs.")


            # --- 3. Calculate Range High/Low (using actual start/end dates) ---
            range_start_dt = pd.Timestamp(f"{actual_start_date_obj} {FIRST_HOUR_START}", tz=TIMEZONE)
            range_end_dt = pd.Timestamp(f"{actual_exp_date_obj} {LAST_HOUR_END}", tz=TIMEZONE)

            if range_end_dt < range_start_dt:
                logging.warning(f"Range end date {actual_exp_date_obj} is before range start date {actual_start_date_obj} for week {week_offset}. Skipping range calculation.")
                range_high, range_low, total_range_val = np.nan, np.nan, np.nan
            else:
                period_data = nifty_df[
                    (nifty_df.index >= range_start_dt) &
                    (nifty_df.index <= range_end_dt)
                ]
                if period_data.empty:
                    logging.warning(f"No Nifty data found in range {range_start_dt} to {range_end_dt}. Using NaNs for range.")
                    range_high, range_low, total_range_val = np.nan, np.nan, np.nan
                else:
                    range_high = period_data[HIGH].max()
                    range_low = period_data[LOW].min()
                    total_range_val = range_high - range_low if pd.notna(range_high) and pd.notna(range_low) else np.nan

            # --- 4. Calculate Derived Features ---
            mon_1h_close = mon_1h_agg.get(f"{MON_FIRST_HOUR_PREFIX}{CLOSE}", np.nan)
            exp_day_close_for_diff = exp_last_hour_close
            exp_day_close_overall = exp_ohlcv.get(f"{EXP_PREFIX}{CLOSE}", np.nan)

            derived_features = pd.Series({
                MON_DATE: mon_date_obj,
                EXP_DATE: actual_exp_date_obj,
                'actual_start_date': actual_start_date_obj,
                WEEK_NUM: week_offset,
                RANGE_HIGH: range_high,
                RANGE_LOW: range_low,
                N_WEEK_HIGH: range_high,
                N_WEEK_LOW: range_low,
                TOTAL_RANGE: total_range_val,
                MON_CLOSE_MINUS_EXP_CLOSE: mon_1h_close - exp_day_close_for_diff if pd.notna(mon_1h_close) and pd.notna(exp_day_close_for_diff) else np.nan,
                N_WEEK_HIGH_MINUS_MON_CLOSE: range_high - mon_1h_close if pd.notna(range_high) and pd.notna(mon_1h_close) else np.nan,
                N_WEEK_LOW_MINUS_MON_CLOSE: range_low - mon_1h_close if pd.notna(range_low) and pd.notna(mon_1h_close) else np.nan,
                N_WEEK_HIGH_MINUS_THURSDAY_CLOSE: range_high - exp_day_close_overall if pd.notna(range_high) and pd.notna(exp_day_close_overall) else np.nan,
                N_WEEK_LOW_MINUS_THURSDAY_CLOSE: range_low - exp_day_close_overall if pd.notna(range_low) and pd.notna(exp_day_close_overall) else np.nan,
            })

            # --- 5. Combine all data for the row ---
            final_row = pd.concat([derived_features, mon_1h_agg, exp_ohlcv, vix_mon_agg, vix_exp_agg])
            processed_data_all_weeks[week_offset].append(final_row)

    # --- Create DataFrames and save (remains the same) ---
    saved_files = {}
    for week_offset, data_list in processed_data_all_weeks.items():
        if not data_list:
            logging.warning(f"No data processed for week {week_offset} expiry.")
            continue
        df_week = pd.DataFrame(data_list)
        df_week = df_week.set_index(MON_DATE)
        df_week.index = pd.to_datetime(df_week.index)
        df_week = df_week.sort_index()
        output_filename = os.path.join(output_dir, f"week{week_offset}_expiry_features.csv")
        try:
            df_week.to_csv(output_filename)
            logging.info(f"Successfully saved processed data to {output_filename}")
            saved_files[week_offset] = output_filename
        except Exception as e:
            logging.error(f"Error saving processed data for week {week_offset} to {output_filename}: {e}")
    logging.info("Weekly expiry data processing finished.")
    return saved_files

def engineer_features(df):
    """Adds technical indicators and other features."""
    logging.info("Engineering additional features...")
    df_out = df.copy()

    # Example: Price Momentum (using Monday 1h close)
    if CALCULATE_MOMENTUM:
        col_mon_close = f"{MON_FIRST_HOUR_PREFIX}{CLOSE}"
        if col_mon_close in df_out.columns:
            for window in ROLLING_WINDOWS:
                 # Simple % change over window
                 df_out[f'{col_mon_close}_mom_{window}d'] = df_out[col_mon_close].pct_change(periods=window) * 100

    # Example: Volatility (using daily range or std dev) - Needs daily data aggregation first
    # For this weekly structure, maybe use Mon 1h high-low or VIX?
    if CALCULATE_VOLATILITY:
         col_mon_high = f"{MON_FIRST_HOUR_PREFIX}{HIGH}"
         col_mon_low = f"{MON_FIRST_HOUR_PREFIX}{LOW}"
         col_vix_close = f"{VIX_PREFIX_MON}{CLOSE}" # Use Monday VIX close

         if col_mon_high in df_out.columns and col_mon_low in df_out.columns:
              df_out['mon_1h_range'] = df_out[col_mon_high] - df_out[col_mon_low]
              for window in ROLLING_WINDOWS:
                   df_out[f'mon_1h_range_vol_{window}d'] = df_out['mon_1h_range'].rolling(window=window).std()

         if col_vix_close in df_out.columns:
              for window in ROLLING_WINDOWS:
                   df_out[f'vix_vol_{window}d'] = df_out[col_vix_close].rolling(window=window).std()


    # Example: Rolling Statistics (using Monday 1h close)
    if CALCULATE_ROLLING_STATS:
         col_mon_close = f"{MON_FIRST_HOUR_PREFIX}{CLOSE}"
         if col_mon_close in df_out.columns:
            for window in ROLLING_WINDOWS:
                df_out[f'{col_mon_close}_roll_mean_{window}d'] = df_out[col_mon_close].rolling(window=window).mean()
                df_out[f'{col_mon_close}_roll_std_{window}d'] = df_out[col_mon_close].rolling(window=window).std()

    # Example: Stationarity Test (ADF) - Apply to relevant features or target diffs
    # Generally applied during analysis rather than feature engineering for all rows
    # Perform this selectively in notebooks or analysis steps.

    # Example: Target Transformation (e.g., for classification)
    if CLASSIFICATION_TARGET_RANGE_BINS in AVAILABLE_CLASSIFICATION_TARGETS and TOTAL_RANGE in df_out.columns:
         # Define bins based on quantiles or fixed values after analysis
         bins = df_out[TOTAL_RANGE].quantile([0, 0.33, 0.66, 1.0]).tolist()
         labels = ['Small', 'Medium', 'Large']
         if len(bins) == 4: # Ensure valid bins
             df_out[CLASSIFICATION_TARGET_RANGE_BINS] = pd.cut(df_out[TOTAL_RANGE], bins=bins, labels=labels, include_lowest=True)
             logging.info(f"Created classification target '{CLASSIFICATION_TARGET_RANGE_BINS}' based on {TOTAL_RANGE} quantiles.")


    df_out.dropna(inplace=True) # Drop rows with NaNs introduced by rolling features/pct_change
    logging.info(f"Finished feature engineering. Shape after dropna: {df_out.shape}")
    return df_out

def perform_stationarity_test(series, col_name=""):
    """Performs ADF test and prints results."""
    if series.isnull().any():
        logging.warning(f"Series '{col_name}' contains NaNs, dropping them for ADF test.")
        series = series.dropna()
    if series.empty:
        logging.warning(f"Series '{col_name}' is empty after dropping NaNs, skipping ADF test.")
        return
    try:
        logging.info(f"Performing ADF test for: {col_name}")
        result = adfuller(series)
        logging.info('ADF Statistic: %f' % result[0])
        logging.info('p-value: %f' % result[1])
        logging.info('Critical Values:')
        for key, value in result[4].items():
            logging.info('\t%s: %.3f' % (key, value))
        if result[1] <= ADFULLER_SIGNIFICANCE_LEVEL:
            logging.info(f"Conclusion: Reject the null hypothesis. Data '{col_name}' is likely stationary.")
        else:
            logging.info(f"Conclusion: Fail to reject the null hypothesis. Data '{col_name}' is likely non-stationary.")
    except Exception as e:
        logging.error(f"Error performing ADF test on '{col_name}': {e}")


# --- Modify process_weekly_expiry_data ---
# ... Inside the function, after the main loop creating df_week ...
# Before saving, engineer features:
# df_week = engineer_features(df_week) # Call the new function

# --- Main Preprocessing Function ---
def run_preprocessing():
    """Loads raw data, processes weekly features, engineers features, and saves."""
    logging.info("--- Running Full Data Preprocessing Pipeline ---")
    nifty_raw = load_data(NIFTY_RAW_FILE)
    vix_raw = load_data(VIX_RAW_FILE)
    if nifty_raw is None or vix_raw is None:
        logging.error("Halting execution due to failure in loading raw data.")
        return None

    processed_files_dict = process_weekly_expiry_data(nifty_raw, vix_raw, PROCESSED_DATA_DIR)

    # Load, engineer features, and re-save each processed file
    final_processed_files = {}
    for week_num, file_path in processed_files_dict.items():
        try:
            logging.info(f"Engineering features for Week {week_num} data...")
            df_proc = pd.read_csv(file_path, index_col=MON_DATE, parse_dates=True)
            if df_proc.index.tz is None: df_proc.index = df_proc.index.tz_localize(TIMEZONE)
            else: df_proc.index = df_proc.index.tz_convert(TIMEZONE)
            df_proc.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential inf

            df_engineered = engineer_features(df_proc) # Apply feature engineering

            # Overwrite the file with the engineered data
            df_engineered.to_csv(file_path)
            logging.info(f"Saved engineered data back to {file_path}")
            final_processed_files[week_num] = file_path
        except Exception as e:
             logging.error(f"Error during feature engineering or re-saving for Week {week_num}: {e}")

    logging.info("--- Full Data Preprocessing Pipeline Finished ---")
    return final_processed_files

# --- Example Usage (__main__ block updated) ---
if __name__ == '__main__':
    logging.info("Running preprocessing module test...")

    raw_nifty_path = NIFTY_RAW_FILE
    raw_vix_path = VIX_RAW_FILE

    # --- Dummy Data Generation (Removed Volume/VolMA) ---
    if not os.path.exists(raw_nifty_path):
        print(f"Creating dummy {raw_nifty_path}")
        base_dates = pd.date_range(start='2023-01-01 09:15:00', end='2023-03-31 15:30:00', freq='BH', tz=TIMEZONE)
        all_hours = []
        for date in base_dates.date:
            start_dt = pd.Timestamp(f"{date} 09:15:00", tz=TIMEZONE)
            end_dt = pd.Timestamp(f"{date} 15:15:00", tz=TIMEZONE)
            all_hours.extend(pd.date_range(start=start_dt, end=end_dt, freq='H'))
        dummy_dates = pd.DatetimeIndex(all_hours)
        dummy_dates = dummy_dates[dummy_dates.date != pd.to_datetime('2023-01-16').date()]
        dummy_dates = dummy_dates[dummy_dates.date != pd.to_datetime('2023-01-26').date()]

        dummy_nifty_data = pd.DataFrame({
            'datetime': dummy_dates,
            'open': np.random.uniform(17000, 18000, len(dummy_dates)),
            'high': lambda x: x['open'] + np.random.uniform(0, 50, len(dummy_dates)),
            'low': lambda x: x['open'] - np.random.uniform(0, 50, len(dummy_dates)),
            'close': lambda x: x['open'] + np.random.uniform(-20, 20, len(dummy_dates)),
            # VOLUME and VOLUME_MA removed
        })
        dummy_nifty_data['high'] = dummy_nifty_data[['high', 'open', 'close']].max(axis=1)
        dummy_nifty_data['low'] = dummy_nifty_data[['low', 'open', 'close']].min(axis=1)
        dummy_nifty_data['datetime'] = dummy_nifty_data['datetime'].dt.strftime(DATETIME_FORMAT.replace("%z", "+05:30"))
        dummy_nifty_data.to_csv(raw_nifty_path, index=False)

    if not os.path.exists(raw_vix_path):
        print(f"Creating dummy {raw_vix_path}")
        # Align VIX dummy data generation (no volume here anyway)
        base_dates_vix = pd.date_range(start='2023-01-01 09:15:00', end='2023-03-31 15:30:00', freq='BH', tz=TIMEZONE)
        all_hours_vix = []
        for date in base_dates_vix.date:
             start_dt = pd.Timestamp(f"{date} 09:15:00", tz=TIMEZONE)
             end_dt = pd.Timestamp(f"{date} 15:15:00", tz=TIMEZONE)
             all_hours_vix.extend(pd.date_range(start=start_dt, end=end_dt, freq='H'))
        dummy_dates_vix = pd.DatetimeIndex(all_hours_vix)
        dummy_dates_vix = dummy_dates_vix[dummy_dates_vix.date != pd.to_datetime('2023-01-16').date()]
        dummy_dates_vix = dummy_dates_vix[dummy_dates_vix.date != pd.to_datetime('2023-01-26').date()]

        dummy_vix_data = pd.DataFrame({
             'datetime': dummy_dates_vix,
             'open': np.random.uniform(10, 25, len(dummy_dates_vix)),
             'high': lambda x: x['open'] + np.random.uniform(0, 2, len(dummy_dates_vix)),
             'low': lambda x: x['open'] - np.random.uniform(0, 2, len(dummy_dates_vix)),
             'close': lambda x: x['open'] + np.random.uniform(-1, 1, len(dummy_dates_vix)),
        })
        dummy_vix_data['high'] = dummy_vix_data[['high', 'open', 'close']].max(axis=1)
        dummy_vix_data['low'] = dummy_vix_data[['low', 'open', 'close']].min(axis=1)
        dummy_vix_data['datetime'] = dummy_vix_data['datetime'].dt.strftime(DATETIME_FORMAT.replace("%z", "+05:30"))
        dummy_vix_data.to_csv(raw_vix_path, index=False)

    # --- Load and Process ---
    nifty_data = load_data(NIFTY_RAW_FILE)
    vix_data = load_data(VIX_RAW_FILE)

    if nifty_data is not None and vix_data is not None:
        process_weekly_expiry_data(nifty_data, vix_data)
    else:
        logging.error("Failed to load necessary data for preprocessing test.")