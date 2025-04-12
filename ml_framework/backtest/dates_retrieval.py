import datetime
import csv
from datetime import date, timedelta

def get_all_mondays(start_year, end_date):
    """
    Generate all Mondays from January 1 of start_year until end_date.
    
    Args:
        start_year (int): The starting year
        end_date (date): The end date to stop generating Mondays
        
    Returns:
        list: A list of all Mondays as date objects
    """
    # Start from January 1st of the start year
    current_date = datetime.date(start_year, 1, 1)
    
    # If January 1st is not Monday (which is 0 in Python's weekday()), 
    # find the first Monday
    while current_date.weekday() != 0:  # 0 is Monday in Python's weekday()
        current_date += datetime.timedelta(days=1)
    
    mondays = []
    
    # Keep adding Mondays until we reach the end date
    while current_date <= end_date:
        mondays.append(current_date)
        current_date += datetime.timedelta(days=7)  # Move to next Monday
    
    return mondays

def save_to_csv(dates, filename):
    """
    Save a list of date objects to a CSV file in yyyy-mm-dd format.
    
    Args:
        dates (list): List of date objects
        filename (str): Name of the CSV file to create
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date'])  # Header
        for date in dates:
            writer.writerow([date.strftime('%Y-%m-%d')])

def main():
    # Starting year is 2021
    start_year = 2021
    
    # Today's date is April 4, 2025
    today = datetime.date(2025, 4, 4)
    
    # Get all Mondays
    mondays = get_all_mondays(start_year, today)
    
    # Save to CSV
    save_to_csv(mondays, 'mondays.csv')
    
    print(f"Successfully saved {len(mondays)} Mondays from 2021-01-01 to {today} in mondays_2021_to_today.csv")

if __name__ == "__main__":
    main()