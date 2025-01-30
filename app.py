import streamlit as st
import requests
import math
import pandas as pd

nse_headers = {
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'DNT': '1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36',
    'Sec-Fetch-User': '?1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-Mode': 'navigate',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
}

def nsefetch(payload):
    try:
        output = requests.get(payload, headers=nse_headers).json()
    except:
        try:
            output = requests.session().get(payload, headers=nse_headers).json()
        except:
            s = requests.session()
            payload2 = "https://www.nseindia.com"
            s.get(payload2, headers=nse_headers)
            output = s.get(payload, headers=nse_headers).json()
    return output

def fetch_nse_data():
    url = "https://www.nseindia.com/api/allIndices"
    data = nsefetch(url)

    nifty_value = None
    vix_value = None

    for index in data['data']:
        if index['index'] == 'NIFTY 50':
            nifty_value = index['last']
        if index['index'] == 'INDIA VIX':
            vix_value = index['last']

    return nifty_value, vix_value

def calculate_range_for_days(nifty_value, vix_value, days):
    years = days / 365
    adjusted_vix = vix_value / math.sqrt(1/years)
    change = nifty_value * (adjusted_vix / 100)

    return {
        'upper': round(nifty_value + change, 2),
        'lower': round(nifty_value - change, 2),
        'range': round(change, 2)
    }

def calculate_nifty_ranges(nifty_value, vix_value):
    results = {}
    yearly_change = nifty_value * (vix_value / 100)
    results['yearly'] = {
        'period': '12-Month',
        'upper': round(nifty_value + yearly_change, 2),
        'lower': round(nifty_value - yearly_change, 2),
        'range': round(yearly_change, 2)
    }

    day_ranges = [7, 15, 30, 45, 60]
    for days in day_ranges:
        key = f'{days}_day'
        results[key] = {
            'period': f'{days}-Day',
            **calculate_range_for_days(nifty_value, vix_value, days)
        }

    return results

# Streamlit UI
st.title('Nifty Range Calculator')

option = st.selectbox('Select Data Source', ('Manual Input', 'Fetch from NSE'))

if option == 'Manual Input':
    nifty_value = st.number_input('Enter Current Nifty Value', min_value=0.0, format='%.2f')
    vix_value = st.number_input('Enter Current India VIX Value', min_value=0.0, format='%.2f')
else:
    nifty_value, vix_value = fetch_nse_data()
    st.write(f'Current Nifty at: {nifty_value}')
    st.write(f'Current India VIX at: {vix_value}')

if nifty_value and vix_value:
    results = calculate_nifty_ranges(nifty_value, vix_value)

    st.header("Nifty VIX Range(s):")

    data = pd.DataFrame.from_dict(results, orient='index')
    st.table(data)
