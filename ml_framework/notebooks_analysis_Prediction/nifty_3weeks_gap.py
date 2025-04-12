import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('/Users/priyeshgupta/sandbox/strategy-data-analysis/nifty-3weeks-expiry/data/NSE_NIFTY_2014.csv')  # Removed relative path for simplicity

# Step 2: Convert the 'time' column to datetime objects, directly handling the format
df['datetime'] = pd.to_datetime(df['time'])
# No need for .dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
#   because the input data already has timezone +05:30 (Asia/Kolkata)

# Step 3: Add a new column for the day of the week
df['day'] = df['datetime'].dt.day_name()

# Step 4: Filter data for the opening candle (9:15-10:15 IST)
df['time_only'] = df['datetime'].dt.time
df = df[(df['time_only'] >= pd.to_datetime('09:15:00').time()) & (df['time_only'] < pd.to_datetime('10:15:00').time())]

# Step 5: Initialize a list to store results
results = []

# Step 6: Iterate through each Monday (or the next available weekday)
# Use iterrows only when necessary; vectorization is far better.
# Here is a vectorized approach.

# Find all Mondays
mondays = df[df['day'] == 'Monday'].copy()

for i in range(len(mondays)):
    monday_data = mondays.iloc[i]
    monday_close = monday_data['close']
    monday_date = monday_data['datetime'].date()

    # Calculate the date for the corresponding Thursday after 3 weeks (17 days)
    target_date = monday_date + pd.Timedelta(days=17)

    # Find the target weekday
    target_data = df[df['datetime'].dt.date == target_date]

    #check target date
    for delta in range(0, -4, -1):  # Check current, Wednesday, Tuesday, or Monday
        target_date_new = target_date + pd.Timedelta(days=delta)
        target_data = df[df['datetime'].dt.date == target_date_new]
        if not target_data.empty:
          break

    if not target_data.empty:
        target_row = target_data.iloc[0]

        #Correct way to compute 3 weeks boundary
        three_weeks_start = monday_data['datetime']
        three_weeks_end = target_row['datetime']

        # Calculate high and low during the 3-week period
        three_weeks_df = df[(df['datetime'] >= three_weeks_start) & (df['datetime'] <= three_weeks_end)]
        three_weeks_high = three_weeks_df['high'].max()
        three_weeks_low = three_weeks_df['low'].min()
        
        # Calculate additional columns
        monday_close_minus_expiry_close = monday_close - target_row['close']
        three_weeks_high_minus_monday_close = three_weeks_high - monday_close
        three_weeks_low_minus_monday_close = three_weeks_low - monday_close
        three_weeks_high_minus_thursday_close = three_weeks_high - target_row['close']
        three_weeks_low_minus_thursday_close = three_weeks_low - target_row['close']

        # Append results
        results.append({
            # Monday data
            'monday_date': monday_date,
            'monday_day': monday_data['day'],
            'monday_open': monday_data['open'],
            'monday_low': monday_data['low'],
            'monday_high': monday_data['high'],
            'monday_close': monday_close,
            'monday_volume': monday_data['Volume'],

            # Expiry day data
            'expiry_date': target_row['datetime'].date(),
            'expiry_day': target_row['day'],
            'expiry_open': target_row['open'],
            'expiry_low': target_row['low'],
            'expiry_high': target_row['high'],
            'expiry_close': target_row['close'],
            'expiry_volume': target_row['Volume'],

            # 3-week high and low
            '3_week_high': three_weeks_high,
            '3_week_low': three_weeks_low,

            # Additional calculations
            'monday_close_minus_expiry_close': monday_close_minus_expiry_close,
            '3_week_high_minus_monday_close': three_weeks_high_minus_monday_close,
            '3_week_low_minus_monday_close': three_weeks_low_minus_monday_close,
            '3_week_high_minus_thursday_close': three_weeks_high_minus_thursday_close,
            '3_week_low_minus_thursday_close': three_weeks_low_minus_thursday_close
        })

# Step 7: Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Step 8: Save the results to a new CSV file
results_df.to_csv('nifty_analysis_results.csv', index=False)

print("Analysis completed and saved to 'nifty_analysis_results.csv'.")