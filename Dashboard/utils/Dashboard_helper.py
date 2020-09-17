"""Dashboard Helper Module"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from urllib.request import urlopen
import datetime
from .predict import SigmoidCurveFit, growth_factor_features, PredictGrowthFactor
from sklearn.metrics import mean_absolute_error

pd.options.plotting.backend = "plotly"

""" Functions for Fetching-Wrangling Data 
"""


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
        # if entry is not of valid format continue to next
        try:
            new_date.append(datetime.datetime.strptime(date + ' 2020', '%d %B %Y'))
        except ValueError:
            continue

    list_dates = new_date

    dataframe = pd.DataFrame(index=list_dates, data=
    {'DailyConfirmed': daily_conf, 'DailyDeceased': daily_dec, 'DailyRecovered': daily_rec,
     'TotalConfirmed': total_conf, 'TotalDeceased': total_dec, 'TotalRecovered': total_rec})

    if save:
        dataframe.to_csv('COVID_India_Updated_from_API.csv')

    return dataframe


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
    # Converting Date string to Datetime

    for rows in data['rows']:
        try:
            date = rows['id'].split('T')[0]
            dates_list.append(datetime.datetime.strptime(date, '%Y-%m-%d'))
            stat_list.append(rows['value']['samples'])
        except ValueError:
            continue

    testing_data = pd.DataFrame(index=dates_list, data={'TestingSamples': stat_list})
    # Removing duplicate indexes
    testing_data = testing_data.loc[~testing_data.index.duplicated(keep='last')]
    testing_data = testing_data.astype('float')

    return testing_data


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
    state_daily_data = state_daily_data.pivot(index='Date', columns='Status')

    if save:
        state_daily_data.to_csv('COVID_India_State.csv')

    return state_daily_data


def country_df(country, df):
    """Filters a Dataframe according to Country.
    Args:
    country: String. Name of country for which dataframe is to be filtered for.
    df: Dataframe. The Dataframe that is to be filtered.
    Returns:
    df_cases: Dataframe. Filtered dataframe containing fields with confirmed Covid cases for the country.
    df_fatal: Dataframe. Filtered dataframe containing fileds with Covid fatalities for the country.
    """
    if country != 'World':
        country_filt = (df['Country/Region'] == country)
        df_cases = df.loc[country_filt].groupby(
            ['Date'])['ConfirmedCases'].sum()
        df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()
    else:
        df_cases = df.groupby(['Date'])['ConfirmedCases'].sum()
        df_fatal = df.groupby(['Date'])['Fatalities'].sum()

    return df_cases, df_fatal


def growth_factor(confirmed):
    confirmed_nminus1 = confirmed.shift(1, axis=0)
    confirmed_nminus2 = confirmed.shift(2, axis=0)
    return (confirmed - confirmed_nminus1) / (confirmed_nminus1 - confirmed_nminus2)


def sharpest_inc_state(data):
    # Group by week and sum, then find which state had the sharpest increase.
    week_state = data.groupby(data.index.week).sum()
    week_state.rename_axis(index='Week', inplace=True)
    # 10 Worst hit states
    state_list = []
    for item in week_state.sort_values(by=week_state.index[-2], axis=1, ascending=False).columns:
        if item[1] == 'Confirmed' and item[0] != 'Total':
            state_list.append(item[0])
    return state_list


""" Loading Data from Different Sources
"""


# fetching data from sqlite db - replace with web versions for deployment with backup CSVs if request times out.

def fetch_data():
    india_data = make_dataframe()

    india_test_series = get_test_dataframe()

    return india_data, india_test_series


""" Functions for returning plots
"""


def log_epidemic_comp(china_cases_df):
    """
    Uses kaggle data to return plots comparing sigmoid and china cases.
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(r"$ \text{Standard logistic sigmoid function} (L=1, k=1, x_0=0) $",
                                        "China cumulative Cases"),
                        vertical_spacing=.1)

    # Plotting a simple logistic curve using numpy and matplotlib.
    # x = (-6,6), L =1, k = 1, x0 =0
    x = np.arange(-6, 7)
    power = -1 * x
    y = 1 / (1 + np.exp(power))
    fig_1 = px.line(x=x, y=y)
    fig_1['data'][0].showlegend = False
    fig.append_trace(fig_1['data'][0], 1, 1)

    # Cases for china
    fig_2 = china_cases_df.plot()
    fig_2['data'][0].showlegend = False
    fig.append_trace(fig_2['data'][0], 1, 2)

    # Updating Axis Titles

    fig.update_yaxes(title_text=r"$f{(x)}$", row=1, col=1)
    fig.update_xaxes(title_text=r"$x$", row=1, col=1)

    fig.update_yaxes(title_text="Confirmed Cases", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=2)

    # figure specs.
    fig.update_layout(title_text="Modelling Epidemic Behaviour",
                      margin=dict(t=55, b=5))
    return fig


def india_national(india_data, value='Daily Statistics'):
    """
    For returning apt plot to Dash app according to radio button.
    """
    if value == 'Daily Statistics':
        fig = india_data[45:][['DailyConfirmed', 'DailyDeceased',
                               'DailyRecovered']].iloc[40:].plot(title=' Daily Trends: India')
        fig.update_layout(yaxis=dict(title='Number of Cases'),
                          xaxis=dict(title='Date'),
                          legend_title_text='', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ), margin=dict(t=10, b=5))
    elif value == 'Weekly Moving Average':
        fig = india_data[45:][['DailyConfirmed', 'DailyDeceased',
                               'DailyRecovered']].rolling(window=7).mean().iloc[40:].plot(
            title=' 7 Day Moving Average : India')
        fig.update_layout(yaxis=dict(title='Number of Cases'),
                          xaxis=dict(title='Date'),
                          legend_title_text='', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ), margin=dict(t=20, b=5))
    elif value == 'Relative Recovery, Death rate':

        # adding death rate and recovery rate.

        recovery_rate = india_data[45:]['DailyRecovered'] / india_data[45:]['DailyConfirmed']
        recovery_rate = recovery_rate.rolling(window=7).mean()
        fig = go.Figure(layout=dict(title='Relative Metrics'))
        fig.add_scatter(y=recovery_rate[40:].values * 100,
                        x=recovery_rate[40:].index.to_list(), mode='lines',
                        name=f'Recovery Rate 7-Day MA (%)',
                        line=dict(
                            color="green", ), )

        death_rate = india_data[45:]['DailyDeceased'] / india_data[45:]['DailyConfirmed']
        death_rate = death_rate.rolling(window=7).mean()
        fig.add_scatter(y=death_rate[40:].values * 100,
                        x=death_rate[40:].index.to_list(), mode='lines',
                        name=f'Death Rate 7-Day MA (%)',
                        line=dict(
                            color="purple", ), )

        fig.update_layout(yaxis=dict(title='Percentage (%) of Confirmed Cases'),
                          xaxis=dict(title='Date'),
                          legend_title_text='', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ), margin=dict(b=5))

    elif value == 'Cumulative Statistics':
        fig = india_data[45:][['TotalConfirmed', 'TotalDeceased',
                               'TotalRecovered']].iloc[40:].plot(title=' Cumulative Trend: India')
        fig.update_layout(yaxis=dict(title='Number of Cases'),
                          xaxis=dict(title='Date'),
                          legend_title_text="", legend=dict(orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02, xanchor="right", x=1),
                          margin=dict(t=20, b=5),
                          )

    return fig


def india_test_plot(india_test, value='Daily Testing and Cases'):
    """
    Plots the test graphs acc. to input from radio button.
    """
    # diff because cumulative test given, we want daily number of tests
    india_test_diff = india_test[['TestingSamples', 'TotalConfirmed']].iloc[40:].diff(1)
    india_test_diff.rename(columns={'TotalConfirmed': 'DailyConfirmed'}, inplace=True)

    if value == 'Daily Testing and Cases':

        fig_test = india_test_diff[50:].plot(title='Daily Testing and '
                                                   'New Cases')
        fig_test.update_layout(yaxis=dict(title='# of Test Samples Collected, Total Confirmed Cases '),
                               legend_title_text='', legend=dict(orientation="h",
                                                                 yanchor="bottom",
                                                                 y=1.02, xanchor="right", x=1),
                               margin=dict(t=10, b=10))

    elif value == 'Moving Average':
        india_test_diff = india_test_diff.rolling(window=7).mean()

        fig_test = india_test_diff[50:].plot(title='Daily Testing and '
                                                   'New Cases 7-Day MA')
        fig_test.update_layout(yaxis=dict(title='# of Test Samples Collected, Total Confirmed Cases'),
                               legend_title_text='', legend=dict(orientation="h",
                                                                 yanchor="bottom",
                                                                 y=1.02, xanchor="right", x=1),
                               margin=dict(t=10, b=10))
    elif value == 'Relative Metrics':
        # NO of sample tested and not people, not a fair comparison. Multiple entries per day, keeping last.

        #  positive rate - no of test returning positive.
        india_test_diff = india_test_diff.rolling(window=7).mean()
        positive_rate = india_test_diff['DailyConfirmed'] / india_test_diff['TestingSamples'] * 100
        positive_rate.rename(index='Positive Rate (%) 7-Day MA', inplace=True)

        fig_test = positive_rate[50:].plot(title='Relative Metrics for Testing')

        fig_test.update_layout(yaxis=dict(title='Percentage (%) of Test Samples, Tests/1000 people'),
                               legend_title_text='', legend=dict(orientation="h",
                                                                 yanchor="bottom",
                                                                 y=1.02, xanchor="right", x=1),
                               margin=dict(t=10, b=10))
        # tests per 1000 people.
        # 1,380,004 - thousands partitions
        test_per_thou = india_test_diff['TestingSamples'] / 1380004
        test_per_thou.rename(index='Tests per thousand people', inplace=True)

        fig_test.add_scatter(y=test_per_thou[50:].values,
                             x=test_per_thou[50:].index.to_list(), mode='lines',
                             name=f'Tests per thousand people 7-Day MA',
                             line=dict(
                                 color="purple", ), )

    fig_test.update_layout(xaxis=dict(title='Date'))

    return fig_test


def national_growth_factor(india_data, value='daily'):
    confirmed = india_data.TotalConfirmed[41:]
    india_growth_factor = growth_factor(confirmed)
    india_growth_factor.rename(index='Growth Factor', inplace=True)
    # for comparison option
    max_val = india_growth_factor[-30:].max()
    min_val = india_growth_factor[-30:].min()
    start = india_growth_factor.index[-30]
    stop = india_growth_factor.index[-1]

    if value == 'Daily Growth Factor' or 'Comparison of Growth Factor':
        # overall growth factor
        fig = india_growth_factor.plot(
            title='India Growth Factor Since 2020-03-11 (Widespread Testing) ')

        # mean growth Factor
        fig.add_scatter(y=[india_growth_factor.mean(), india_growth_factor.mean()],
                        x=[india_growth_factor.index[0], india_growth_factor.index[-1]], mode='lines',
                        name=f'Mean Growth Factor = {india_growth_factor.mean():.4f}',
                        line=dict(
                            color="crimson",
                            width=4,
                            dash="dashdot",
                        ),
                        )
        # growth factor for last month
        fig.add_scatter(y=[india_growth_factor[-30:].mean(), india_growth_factor[-30:].mean()],
                        x=[india_growth_factor.index[-30], india_growth_factor.index[-1]], mode='lines',
                        name=f'Mean Growth Factor Last 30 days = {india_growth_factor[-30:].mean():.4f}',
                        line=dict(
                            color="darkviolet",
                            width=4,
                            dash="dashdot",
                        ),
                        )
        # growth factor for last 7 days
        fig.add_scatter(y=[india_growth_factor[-7:].mean(), india_growth_factor[-7:].mean()],
                        x=[india_growth_factor.index[-7], india_growth_factor.index[-1]], mode='lines',
                        name=f'Mean Growth Factor Last 7 days = {india_growth_factor[-7:].mean():.4f}',
                        line=dict(
                            color="lightseagreen",
                            width=4,
                            dash="dashdot",
                        ),
                        )

        # Updating legend location
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ), legend_title_text='',
            margin=dict(
                b=20,
                pad=1
            ))
        fig.update_yaxes(title_text="Growth Factor")
        fig.update_xaxes(title_text="Date")
    if value == 'Growth Factor Weekly Moving Average':
        x_ma = india_growth_factor[2:].rolling(window=7).mean()[6:]
        fig = x_ma.plot(title='India Weekly Moving Average Growth Factor Since 2020-03-11 (Widespread Testing) ')
        # Updating legend location
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ), legend_title_text='',
            margin=dict(
                b=20,
                pad=1
            ),
        )
        fig.add_scatter(y=[1, 1],
                        x=[x_ma.index[0], x_ma.index[-1]], mode='lines',
                        name=f'Inflection Point',
                        line=dict(
                            color="crimson",
                            width=4,
                            dash="dashdot",
                        ),
                        )
        fig.update_yaxes(title_text="Growth Factor")
        fig.update_xaxes(title_text="Date")

    return fig, min_val, max_val, start, stop


def state_plots(state_series, state_name):
    # Creating subplots
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(f"Daily Trends", f"7-Day Moving Average",
                                        f"Cumulative Cases", f"Growth Factor 7-Day Moving Average"),
                        vertical_spacing=.1, )
    # 1. Daily Trends
    fig_1 = state_series.plot()
    # adding figure to subplot
    fig.append_trace(fig_1['data'][0], 1, 1)
    fig.append_trace(fig_1['data'][1], 1, 1)
    fig.append_trace(fig_1['data'][2], 1, 1)

    # 2. Weekly Moving Average
    fig_2 = state_series.rolling(window=7).mean().plot()
    # removing redundant legends
    fig_2['data'][0].showlegend = False
    fig_2['data'][1].showlegend = False
    fig_2['data'][2].showlegend = False

    fig.append_trace(fig_2['data'][0], 1, 2)
    fig.append_trace(fig_2['data'][1], 1, 2)
    fig.append_trace(fig_2['data'][2], 1, 2)

    # 3. Cumulative Cases
    fig_3 = state_series.cumsum().plot()

    fig_3['data'][0].showlegend = False
    fig_3['data'][1].showlegend = False
    fig_3['data'][2].showlegend = False

    fig.append_trace(fig_3['data'][0], 2, 1)
    fig.append_trace(fig_3['data'][1], 2, 1)
    fig.append_trace(fig_3['data'][2], 2, 1)

    # 4. Weekly rolling growth factor
    gf_df = growth_factor(state_series['Confirmed'].cumsum())
    gf_df.rename(index='Growth Factor', inplace=True)
    fig_4 = gf_df.rolling(window=7).mean().plot()
    fig_4['data'][0].line['color'] = "black"
    fig.append_trace(fig_4['data'][0], 2, 2)

    # Axis Titles
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=2)
    fig.update_yaxes(title_text="Cases", row=2, col=1)
    fig.update_yaxes(title_text="Growth Factor", row=2, col=2)

    # figure specs.
    fig.update_layout(height=700, showlegend=True,
                      title_text=f"{state_name} Statistics", legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="right",
            x=1
        ))

    return fig


def forecast_curve_fit(india_data, x_data, y_data):
    # dict to store eval metrics
    score = {}

    # splitting into train and test(last 30 days) without shuffling.
    index = 30
    x_train = x_data[:-index]
    x_test = x_data[-index:]
    y_train = y_data[:-index]
    y_test = y_data[-index:]

    # Instantiating and fitting logistic function
    sigmoid = SigmoidCurveFit()
    sigmoid.fit(x_train.reshape(-1, 1), y_train)

    score['R^2'] = sigmoid.score(x_test, y_test)
    score['MAE'] = mean_absolute_error(y_test, sigmoid.predict(x_test))
    score['params'] = sigmoid.get_sigmoid_params()

    # make everything other than every 5th value nan
    df = india_data.copy()
    df['TotalConfirmed'][~df.index.isin(df[::5].index)] = np.nan

    # preds for only test set
    preds_test = sigmoid.predict(x_test.reshape(-1, 1))

    # predict for entire input
    preds = sigmoid.predict(x_data.reshape(-1, 1))

    # figure for train-valid fit

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Logistic function fit to Cumulative Cases",
                                        "Validation Set Predictions for Daily Cases"),
                        column_widths=[0.6, 0.4],
                        horizontal_spacing=0.08)

    fig_1 = df.rename(columns={'TotalConfirmed': 'ActualCases'})['ActualCases'].plot(kind='scatter')

    fig_1.update_traces(
        line=dict(dash="dot", width=10)
    )

    fig.append_trace(fig_1['data'][0], 1, 1)

    fig.update_traces(marker=dict(color="#636EFA"),
                      row=1, col=1)

    fig.add_scatter(x=df.index, y=preds.ravel(),
                    name='Fit Line',
                    line=dict(
                        color="#EF553B",
                        width=2),
                    marker=dict(
                        opacity=0.4)
                    )

    # second figure - index is 29 days as 1 day differenced away

    fig_2 = px.line(x=india_data.index[-29:], y=np.diff(y_test, n=1), title="ActualCases")
    fig_2.add_scatter(x=india_data.index[-29:], y=np.diff(preds_test.ravel(), n=1), mode='lines',
                      name='Predicted Cases',
                      line=dict(
                          color="#EF553B"),
                      )
    fig_2['data'][1].showlegend = False
    fig.append_trace(fig_2['data'][0], 1, 2)
    fig.append_trace(fig_2['data'][1], 1, 2)

    fig.update_layout(shapes=[
        dict(
            type='line',
            yref='paper', y0=0, y1=1,
            xref='x', x0=df.index[-30], x1=df.index[-30]
        )
    ], legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    ), legend_title_text='',
        margin=dict(
            b=15,
            t=5,
            l=50,
            r=50,
            pad=1
        ),
    )
    # Axis Titles
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=2)

    return fig, score


def forecast_growth_factor(india_data):
    # Creating growth factor features
    confirmed = india_data.TotalConfirmed[41:]
    india_growth_factor = growth_factor(confirmed)
    feat_df = india_growth_factor.iloc[2:].copy()
    feat_df = feat_df.to_frame(name='Growth_Factor')
    data = growth_factor_features(feat_df)

    # drop first 6 na values in lag of GF
    data.dropna(inplace=True)

    # validation set approach - No shuffle time sries data. Last 30 days removed
    z = 30
    x_train = data.drop(['Growth_Factor'], axis=1)[:-z]
    x_test = data.drop(['Growth_Factor'], axis=1)[-z:]
    y_train = data['Growth_Factor'][:-z]
    y_test = data['Growth_Factor'][-z:]

    # Create Object of SuperPredictor Regression and get predictions
    estimators = PredictGrowthFactor()
    estimators.fit(x_train, y_train)

    # dictionary of results
    preds = estimators.predict(x_test)

    temp = pd.DataFrame(index=x_test.index, data=y_test)
    fig = temp.plot()
    fig.add_scatter(x=x_test.index, y=preds['lin_reg'], name='Linear Regression')
    fig.add_scatter(x=x_test.index, y=preds['last_month_mean'], name='Mean Last Month')
    fig.add_scatter(x=x_test.index, y=preds['ridge_reg_deg_2'], name='Ridge Regression')

    return fig, preds
