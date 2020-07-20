```python
"""
Querying a public API for JSON data
Using Covid19india.org api to get daily cases, states, etc.

Data Format: JSON
data in the format of ['cases_time_series'] with objects for each day nested in it. ['key_values']
another key/object in it of the form ['statewise']

Data starting from 30th January to the current day.
### TO DO
1. Other functions for statewise data etc.

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
response = urlopen("https://api.covid19india.org/v4/timeseries.json")
source = response.read()
data = pd.read_json(source, orient='records')
```

```python
response = urlopen('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
source = response.read()
```

```python
pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv', index_col=0)
```

```python
pd.read_json(source, orient='index')
```

```python
pd.json_normalize(json.loads(source), max_level=2)
```

```python
from IPython.display import display
```

```python
import numpy as np
```

```python
#pd.read_json(json.dumps(data['AN']), orient='records')
```

```python
with urlopen("https://api.covid19india.org/v4/timeseries.json") as response:
    source = response.read()
    pd.read_json(source)
    data = json.loads(source)
    #print(data['DL']['dates'].keys())
    daily_ts = pd.read_json(json.dumps(data['DL']['dates']), orient='index')
    daily_ts.fillna('{\'confirmed\': 0}', inplace=True)
    display(daily_ts)
    #df['stats'].apply(json.loads)
    #pd.json_normalize(json.dumps(daily_ts['delta'].to_dict()))
    #pd.json_normalize(daily_ts['delta'].to_dict())

    

    #display(pd.json_normalize(json.loads(json.dumps(data['DL']['dates'])), max_level=4))
    # total time series and delta time_series 
    
    #for state in state_list:
        # now the nested object has a key of dates 
        #display(pd.DataFrame(data['DL']))
        #list_cases_stat(data['DL'], 'dailyconfirmed')
```

```python
pd.json_normalize(data['DL']['dates'], max_level=2)
```

```python
#pd.json_normalize([daily_ts.delta.to_dict()])
```

```python
[daily_ts.delta.to_dict()]
```

```python

```

```python
def make_dataframe(save=False):
    """"Makes Dataframe with parsed data and and returns it with an option to save it as a CSV.
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
def get_test_dataframe():
    """Gets ICMR covid Testing samples data from datameet dataset.
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
