import pandas as pd
import ast
import numpy as np # Import numpy for NaN handling

# === Load Data ===
file_path = 'backtest/results/model_results/backtest_w4_highlow_custom.csv'  # Replace this with your CSV path
df = pd.read_csv(file_path)

# === Define Actual Column Names ===
# !!! IMPORTANT: Adjust these if your column names are different !!!
actual_high_col = 'actual_high'
actual_low_col = 'actual_low'
actual_close_col = 'actual_close' # <<< This column MUST exist in your CSV

# Check if actual_close column exists
if actual_close_col not in df.columns:
    raise ValueError(f"Error: The required column '{actual_close_col}' was not found in the CSV.")

# === Prediction Columns (with brackets) ===
high_model_cols = [
    'predictions_n_week_high_individual_RandomForestRegressor_value',
    'predictions_n_week_high_individual_LGBMRegressor_value',
    'predictions_n_week_high_individual_CatBoostRegressor_value',
    'predictions_n_week_high_individual_XGBRegressor_value'
]

low_model_cols = [
    'predictions_n_week_low_individual_RandomForestRegressor_value',
    'predictions_n_week_low_individual_LGBMRegressor_value',
    'predictions_n_week_low_individual_CatBoostRegressor_value',
    'predictions_n_week_low_individual_XGBRegressor_value'
]

# === Helper to clean and parse bracketed values ===
def parse_single_value(val):
    try:
        if pd.isna(val):
            return np.nan # Use numpy's NaN for consistency
        # Handle potential strings like '[123.45]' or just numbers
        if isinstance(val, str):
            parsed = ast.literal_eval(val)
            # Expecting a list/tuple with one number, like '[150.0]'
            if isinstance(parsed, (list, tuple)) and len(parsed) == 1:
                 return float(parsed[0])
            else:
                 # Handle cases where literal_eval might return a number directly
                 # or if the string format is unexpected
                 return float(parsed) # Try converting directly if not list/tuple
        else:
            return float(val) # It might already be a number
    except (ValueError, SyntaxError, TypeError):
        # Handle various errors during parsing or conversion
        return np.nan

# === Clean and convert all model columns ===
for col in high_model_cols + low_model_cols:
    df[col] = df[col].apply(parse_single_value)
    # Ensure numeric type after parsing, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')


# === Best-case predictions ===
# Use fillna(np.inf) for min and fillna(-np.inf) for max
# so that if all predictions are NaN, the result is NaN,
# but if only some are NaN, the valid numbers are used.
df['n_week_high_pred'] = df[high_model_cols].max(axis=1, skipna=True)
df['n_week_low_pred'] = df[low_model_cols].mean(axis=1, skipna=True)

# === Reliability Scoring Function ===
def calculate_reliability_score(row):
    """Calculates reliability score based on closing price and breaches."""
    pred_low = row['n_week_low_pred']
    pred_high = row['n_week_high_pred']
    actual_low = row[actual_low_col]
    actual_high = row[actual_high_col]
    actual_close = row[actual_close_col]

    # Handle cases where necessary data is missing
    if pd.isna(pred_low) or pd.isna(pred_high) or pd.isna(actual_low) or pd.isna(actual_high) or pd.isna(actual_close):
        return np.nan # Cannot calculate score if data is missing

    score = 0
    closed_within = (actual_close >= pred_low) and (actual_close <= pred_high)
    high_broken = actual_high > pred_high
    low_broken = actual_low < pred_low

    # Rule 1: Close within range (Positive Point)
    if closed_within:
        score += 1
        # Rule 2: High broken, closed within range (Negative Point)
        if high_broken:
            score -= 1
        # Rule 3: Low broken, closed within range (Negative Point)
        if low_broken:
            score -= 1
        # Rule 4 (Implicit): Both broken, closed within -> handled by Rule 2 + Rule 3 (-1 + -1 = -2)

    # Rule 5: High broken AND closed ABOVE range (Two Negative Points)
    elif high_broken and actual_close > pred_high: # Check close > pred_high specifically
        score -= 2
        # Does low break matter if it closed high? User didn't specify extra penalty.

    # Rule 6: Low broken AND closed BELOW range (Two Negative Points)
    elif low_broken and actual_close < pred_low: # Check close < pred_low specifically
        score -= 2
        # Does high break matter if it closed low? User didn't specify extra penalty.

    # Consider edge case: Not closed within, but high/low not broken?
    # e.g., actual range is entirely outside predicted, but close is outside too.
    # Example: pred [100, 110], actual [112, 115], close 113. high_broken=True, close>pred_high=True -> Score = -2 (Rule 5) Correct.
    # Example: pred [100, 110], actual [95, 98], close 97. low_broken=True, close<pred_low=True -> Score = -2 (Rule 6) Correct.

    return score

# === Apply Scoring ===
df['reliability_score'] = df.apply(calculate_reliability_score, axis=1)

# === Calculate Cumulative Score ===
# Fill NaN scores with 0 before calculating cumulative sum, or decide how to handle gaps
df['cumulative_reliability_score'] = df['reliability_score'].fillna(0).cumsum()

# === Styling Function for Visualization ===
def style_reliability(score):
    """Applies background color based on reliability score."""
    if pd.isna(score):
        return 'background-color: grey'
    elif score > 0:
        return 'background-color: lightgreen'
    elif score == 0:
        return 'background-color: lightyellow'
    elif score == -1:
        return 'background-color: lightcoral'
    elif score <= -2:
        return 'background-color: crimson; color: white' # More intense red for bigger penalties
    else:
        return '' # Default

# === Prepare DataFrame for Display ===
# Select relevant columns (adjust 'test_date' if named differently)
display_cols = [
    'test_date', # Assuming 'test_date' or similar exists for context
    actual_high_col,
    'n_week_high_pred',
    'n_week_low_pred',
    actual_low_col,
    actual_close_col, # Show the crucial close price
    'reliability_score'
]

# Ensure 'test_date' exists or replace it
if 'test_date' not in df.columns:
     if 'Date' in df.columns:
         display_cols[0] = 'Date'
     elif 'index' in df.columns: # Fallback to index if no date
         display_cols[0] = 'index'
     else: # If no date/index, just remove it from display list
         display_cols.pop(0)
         df = df.reset_index() # Ensure we have an index column if needed


df_display = df[display_cols].copy() # Work on a copy for styling

# === Apply Styling and Display ===
# Note: Styling is best viewed in environments like Jupyter notebooks/lab
# It won't be saved in the CSV output.
styled_df = df_display.style.applymap(style_reliability, subset=['reliability_score'])

print("\n=== Reliability Score Analysis (Styled) ===")
# Display the styled DataFrame (might be large, consider df_display.head(20).style... for brevity)
# In some environments, you might just need to have `styled_df` as the last line.
try:
    from IPython.display import display
    display(styled_df)
except ImportError:
    # Fallback for non-notebook environments (won't show colors)
    print("(Styling requires IPython/Jupyter environment to display colors)")
    print(df_display)


# === Save output with new scores ===
# The CSV will contain the scores, but not the colors.
output_path = 'backtest/results/failure_rate/Week4_processed_predictions_reliability_scores.csv'
# Add cumulative score to the output df
df_to_save = df[display_cols + ['cumulative_reliability_score']].copy()
df_to_save.to_csv(output_path, index=False)
print(f"\nDataFrame with reliability scores saved to: {output_path}")

# === Optional: Print Summary Statistics ===
print("\n=== Score Summary ===")
if not df['reliability_score'].isna().all(): # Check if there are any valid scores
    print(df['reliability_score'].value_counts().sort_index())
    total_positive_points = df[df['reliability_score'] > 0]['reliability_score'].sum()
    total_negative_points = df[df['reliability_score'] < 0]['reliability_score'].sum()
    print(f"\nTotal Positive Points: {total_positive_points}")
    print(f"Total Negative Points: {total_negative_points}")
    print(f"Overall Net Score: {df['reliability_score'].sum()}")
    print(f"Final Cumulative Score: {df['cumulative_reliability_score'].iloc[-1]}") # Last value of cumulative
else:
    print("No valid reliability scores could be calculated (check input data and predictions).")