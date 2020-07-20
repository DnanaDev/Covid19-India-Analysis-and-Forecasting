```python
"""
Querying a public API for Covid
Using Covid19india.org api to get daily cases, states, etc.

Data Format: JSON
data in the format of ['cases_time_series'] with objects for each day nested in it. ['key_values']
another key/object in it of the form ['statewise']

Data starting from 30th January to the current day.


"""
```

```python
import json
from urllib.request import urlopen
import pandas as pd
import datetime
```


got the json data but it is not very readable
print(source)


print(json.dumps(data, indent=2))


data in the format of ['cases_time_series'] with objects for each day nested in it. ['key_values']
another key/object in it of the form ['statewise']
Lets read cases time-series
"dailydeceased": "0",
"dailyrecovered": "0",
"date": "30 January ",
"totalconfirmed": "1",
"totaldeceased": "0",
"totalrecovered": "0"



print(json.dumps(data['cases_time_series'], indent = 2))


```python
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
```

```python
def make_dataframe(save=False):
    """"Makes Dataframe with parsed data and and returns it with an option to save it as a CSV.
    Data starting - 2020-01-30.
    Args:
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

    if save:
        dataframe.to_csv('COVID_India_Updated_from_API.csv')

    return dataframe
```

```python
def make_state_dataframe(save=False):
    """Returns Dataframe with parsed data for national and statewise cases timeseries.
    Optional to save CSV. Data starting - 2020-03-14.
    Args:
    save: Saves the cleaned CSV.
    Returns:Dataframe. With stacked Columns DailyConfirmed, DailyDeceased, DailyRecovered
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
    state_daily_data = state_daily_data.pivot(index = 'Date',columns = 'Status')
   
    return state_daily_data
```

```python
def get_test_dataframe():
    """Gets ICMR covid Testing samples data from datameet dataset.
    Data starting - 2020-03-13.
    Args:
    Returns:
    Dataframe with Date, Number of samples collected on that day.
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

    testing_data = pd.DataFrame(index=dates_list, data={'Testing Samples': stat_list})

    # Converting Date string to Datetime
    dates = []
    for date in testing_data.index.to_list():
        dates.append(datetime.datetime.strptime(date, '%Y-%m-%d'))

    testing_data.index = dates
    # testing_data.to_csv('COVID_India_Updated_Test_data.csv')

    return testing_data
```
