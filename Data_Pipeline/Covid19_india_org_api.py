"""
## Querying Covid19india.org API for Daily Covid Data.
Using Covid19india.org api to fetch daily confirmed, recovered, deceased case numbers for each state, and
 daily testing samples collected for the entire nation.

## Getting Started
Functions:
1. make_dataframe(save = False)
Returns DataFrame with data parsed from source [1].
National time-series with Daily/TotalConfirmed, Daily/TotalDeceased, Daily/TotalRecovered
Data starting - 2020-01-30.
Args : save - Path to save the cleaned CSV. Default False to not save file.


2. make_state_dataframe(save = False)
Returns Multi-indexed DataFrame with data parsed from source [2].
*Version 2* Returns Long CSV in place of Wide CSV - State is a column now.
National and State wise time-series with DailyConfirmed, DailyDeceased, DailyRecovered
Data starting - 2020-03-14.
Args : save - Path to save the cleaned CSV. Default False to not save file.

3. get_test_dataframe(save = False)
Returns DataFrame with data parsed from source [3].
National Testing time-series. Multiple entries exist for particular dates.
Data starting - 2020-03-13.
Args : save - Path to save the cleaned CSV. Default False to not save file.

## Data Sources

1. Covid19india.org API Daily National time-series JSON
https://api.covid19india.org/data.json

Data Format: JSON
data in the format of ['cases_time_series'] with objects for each day nested in it. ['key_values']
another key/object in it of the form ['statewise']
Data starting from 30th January to the current day.
# got the json data but it is not very readable
# print(source)
# print(json.dumps(data, indent=2))
# data in the format of ['cases_time_series'] with objects for each day nested in it. ['key_values']
# another key/object in it of the form ['statewise']
# Lets read cases time-series
# "dailydeceased": "0",
# "dailyrecovered": "0",
# "date": "30 January ",
# "totalconfirmed": "1",
# "totaldeceased": "0",
# "totalrecovered": "0"
# print(json.dumps(data['cases_time_series'], indent = 2))

2. Covid19india.org API Daily State wise time-series CSV
https://api.covid19india.org/csv/latest/state_wise_daily.csv

Data Format: CSV
Perfectly clean CSV. Using source [1] to rename state code columns to State name (consistent with API, makes it more
readable)
Data starting 2020-03-14.

3. Datameet Daily ICMR Covid Testing samples data JSON
https://raw.githubusercontent.com/datameet/covid19/master/data/icmr_testing_status.json

Scraped Testing data from ICMR notices.
Data starting - 2020-03-13.
Multiple entries exist for particular dates.

## TO DO
1. Make pipeline suitable for GCS bucket and Google Cloud Function.

**** All Data sources reachable and working as of - 20.07.20 ***
"""

import json
from urllib.request import urlopen
import pandas as pd
import datetime


def list_cases_stat(json_obj, column):
    """
    This function takes in the JSON retrieved from the API and returns a list of integers for the specified
    column, can be used to then plot the time-series.
    :param json_obj: The JSON object returned by the API.
    :param column: The metric that is to be returned, dailyconfirmed etc.
    :return: Returns a list of integers of the time series for the specified stat.
    """
    stat_list = []
    for daily_num in json_obj['cases_time_series']:
        stat_list.append(daily_num[column])
    try:
        stat_list_int = [int(x) for x in stat_list]
    except ValueError:
        stat_list_int = stat_list

    return stat_list_int


def make_dataframe(save=False):
    """"Makes Dataframe with parsed data and and returns it with an option to save it as a CSV.
    Data starting - 2020-01-30.
    Args:
    save: Path to save the cleaned CSV. Default False to not save file.
    Returns:Dataframe. With Columns DailyConfirmed, DailyDeceased, DailyRecovered."""

    # Fetching The JSON
    with urlopen("https://api.covid19india.org/data.json") as response:
        source = response.read()
        data = json.loads(source)

    # Getting Data From Json Object using list_cases_stat function
    daily_conf = list_cases_stat(data, 'dailyconfirmed')
    daily_dec = list_cases_stat(data, 'dailydeceased')
    daily_rec = list_cases_stat(data, 'dailyrecovered')
    total_conf = list_cases_stat(data, 'totalconfirmed')
    total_dec = list_cases_stat(data, 'totaldeceased')
    total_rec = list_cases_stat(data, 'totalrecovered')

    list_dates = list_cases_stat(data, 'date')

    # Converting Dates to 'datetime'
    new_date = []
    for date in list_dates:
        new_date.append(datetime.datetime.strptime(date + ' 2020', '%d %B %Y'))

    list_dates = new_date

    dataframe = pd.DataFrame(index=list_dates, data=
    {'DailyConfirmed': daily_conf, 'DailyDeceased': daily_dec, 'DailyRecovered': daily_rec,
     'TotalConfirmed': total_conf, 'TotalDeceased': total_dec, 'TotalRecovered': total_rec})
    # Renaming Index to be consistent with all other CSVs
    dataframe.rename_axis(index = 'Date', inplace=True)

    if save:
        dataframe.to_csv(save)

    return dataframe


def make_state_dataframe(save=False):
    """Returns Dataframe with parsed data for national and statewise cases timeseries.
    Optional to save CSV. Data starting - 2020-03-14.
    Args:
    save: Path to save the cleaned CSV. Default False to not save file.
    Returns:Dataframe. With Columns State, DailyConfirmed, DailyDeceased, DailyRecovered
    for each state and Total - National time series.
    """

    # Dictionary for renaming state codes to full state names, slightly wasteful, 
    # additional API call to different file.
    response = urlopen("https://api.covid19india.org/data.json")
    source = response.read()
    data = json.loads(source)

    # creating dict for pandas DF rename mapper.
    state_identifier = {}
    for record in data['statewise']:
        state_identifier[record['statecode']] = record['state']

    # Read in CSV, rename, pivot to make datetime index
    state_daily_data = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    state_daily_data.rename(columns=state_identifier, inplace=True)
    state_daily_data.Date = pd.to_datetime(state_daily_data.Date)
    state_daily_data = state_daily_data.pivot(index='Date', columns='Status')

    # Melt stacked Column index Dataframe to long Dataframe
    state_daily_data.reset_index(inplace=True)
    state_daily_data = state_daily_data.melt(id_vars=['Date'])

    # renaming orphan column with state data
    state_daily_data.rename(axis=1, mapper={None: 'State'}, inplace=True)

    # Pivoting to reshape Status column values(Recovered, Confirmed, Deceased Cases) to columns
    state_daily_data = state_daily_data.pivot_table(index=['Date', 'State'], columns='Status', values='value')

    # Reset index to replicate stacked index Date to Date column before setting as index.
    state_daily_data = state_daily_data.reset_index().set_index('Date')

    if save:
        state_daily_data.to_csv(save)

    return state_daily_data


def get_test_dataframe(save=False):
    """Gets ICMR Covid Testing samples data from Datameet dataset.
    Data starting - 2020-03-13. Has multiple entries for certain days.
    Args:
    save: Path to save the cleaned CSV. Default False to not save file.
    Returns:
    DataFrame with Date, Number of samples collected on that day.
    """
    path_testing = 'https://raw.githubusercontent.com/datameet/covid19/master/data/icmr_testing_status.json'

    with urlopen(path_testing) as response:
        # Reading this json data
        source = response.read()
        # converting this json to
        data = json.loads(source)

    stat_list = []
    dates_list = []

    # Parsing Dates and Number of Samples Collected on day.
    for rows in data['rows']:
        dates_list.append(rows['id'].split('T')[0])
        stat_list.append(rows['value']['samples'])

    testing_data = pd.DataFrame(index=dates_list, data={'TestingSamples': stat_list})

    # Converting Date string to Datetime
    dates = []
    for date in testing_data.index.to_list():
        dates.append(datetime.datetime.strptime(date, '%Y-%m-%d'))

    testing_data.index = dates
    # Renaming Index to be consistent with all other CSVs
    testing_data.rename_axis(index='Date', inplace=True)

    if save:
        testing_data.to_csv(save)

    return testing_data


if __name__ == '__main__':
    # When run directly, saves updated data from all three sources.
    make_dataframe(save='../Data/Raw/COVID_India_National.csv')
    make_state_dataframe(save='../Data/Raw/COVID_India_State.csv')
    get_test_dataframe(save='../Data/Raw/COVID_India_Test_data.csv')
