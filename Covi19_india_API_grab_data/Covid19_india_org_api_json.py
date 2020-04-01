"""
Querying a public API for JSON data
Using Covid19india.org api to get daily cases, states, etc.
"""

import json
from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd

with urlopen("https://api.covid19india.org/data.json") as response:
    source = response.read()

# got the json data but it is not very readable
# print(source)

data = json.loads(source)


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


# Data for 62 days, starting from 30th January to the current day.

# print(json.dumps(data['cases_time_series'], indent = 2))


# Create function for lines 40 - 53

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


daily_conf = list_cases_stat(data, 'dailyconfirmed')
daily_dec = list_cases_stat(data, 'dailydeceased')
daily_rec = list_cases_stat(data, 'dailyrecovered')

# Now plotting Daily Figures

plt.style.use('seaborn')
plt.title('Daily Trends: India')
plt.ylabel('Daily Growth in Cases')
plt.xlabel('Days since 30th January')
plt.plot(daily_conf, label='Daily Cases')
plt.plot(daily_dec, label='Daily Deceased')
plt.plot(daily_rec, label='Daily Recovered')
plt.legend()
plt.savefig('India_Daily_Stats.png')
plt.show()

# Now Lets plot Total Cases

total_conf = list_cases_stat(data, 'totalconfirmed')
total_dec = list_cases_stat(data, 'totaldeceased')
total_rec = list_cases_stat(data, 'totalrecovered')

fig, ax = plt.subplots()
ax.plot(total_conf, label='Total Confirmed Cases')
ax.plot(total_dec, label='Total Deceased')
ax.plot(total_rec, label='Total Recovered')
ax.set_title('Cumulative Stats : India')
ax.set_xlabel('Days since 30th January')
ax.set_ylabel('Count')
fig.legend()
fig.savefig('India_Cumulative_stats.png')
fig.show()

# Lets get the dates

list_dates = list_cases_stat(data, 'date')

# Next lets automatically store this data in a Pandas Dataframe

dataframe = pd.DataFrame(
    {'Date': list_dates, 'DailyConfirmed': daily_conf, 'DailyDeceased': daily_dec, 'DailyRecovered': daily_rec,
     'TotalConfirmed': total_conf, 'TotalDeceased': total_dec, 'TotalRecovered': total_rec})

dataframe.to_csv('COVID_India_Updated_from_API.csv', index=False)
