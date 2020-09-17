# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_defer_js_import as dji
import pandas as pd
import numpy as np

# import functions and data from helper function

from utils.Dashboard_helper import india_national, log_epidemic_comp, country_df, india_test_plot, \
    national_growth_factor, make_state_dataframe, sharpest_inc_state, state_plots, forecast_curve_fit, \
    forecast_growth_factor, fetch_data

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

###### important for latex ######
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
'''

###### important for latex ######
axis_latex_script = dji.Import(src="https://codepen.io/yueyericardo/pen/pojyvgZ.js")
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")

""" Data sources, Wrangling
"""

### Fetching Data ###
kaggle_snap_worldwide = pd.read_csv('utils/Data/kaggle_train.csv')
kaggle_snap_worldwide.rename(columns={'Country_Region': 'Country/Region',
                                      'Province_State': 'Province/State'}, inplace=True)

india_data, india_test_series = fetch_data()

state_data = make_state_dataframe()

### wrangling Data ###
china_cases, china_fatal = country_df('China', kaggle_snap_worldwide)

states = sharpest_inc_state(state_data)

india_data_combined = india_data.join(india_test_series, how='left')

# making copy for test data functions
india_test = india_data_combined.copy()

# Data for Forecasting Models

# For Total Cases
# output is the number of cumulative/total cases
# Input - days since first reported case, 2020-01-30

x_data = np.arange(len(india_data['TotalConfirmed']))

y_data = india_data['TotalConfirmed'].values

### Figures ###

fig_china = log_epidemic_comp(china_cases)

fig_growth_factor = national_growth_factor(india_data)

""" App layout
"""
app.layout = html.Div([
    # Overall header
    html.H1(children='Covid-19 : India Analysis and Forecasting', style={'textAlign': 'center', 'margin-top': '20px'}),
    # header
    html.Div([html.P(['''Dashboard for analysing and forecasting the spread of the pandemic using Epidemiological,
    Time-Series and Machine Learning models and metrics.''',
                      html.Br(), html.B('Disclaimer : '),
                      '''The authors of the dashboard are not epidemiologists and any analysis and forecasting is not be 
                      taken seriously. The statistics, metrics and forecasts update daily and the analysis was performed
                       in the first week of September.'''],
                     style={'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
              ]),

    # Div - Nationwide Analysis
    html.Div([
        html.H4('Nation-wide Statistics', style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
                                                 'margin-right': '15px'}),
        dcc.Graph(
            id='national-stats'),
        html.Div([dcc.Dropdown(id='nat-graph-selector',
                               options=[{'label': i, 'value': i} for i in ['Daily Statistics', 'Cumulative Statistics',
                                                                           'Weekly Moving Average',
                                                                           'Relative Recovery, Death rate']],
                               value='Daily Statistics', style={'width': '50%',
                                                                'align-items': 'left', 'justify-content': 'center',
                                                                'margin-left': '10px', 'margin-right': '15px'}),
                  dcc.RadioItems(
                      options=[
                          {'label': 'Linear', 'value': 'linear'},
                          {'label': 'Log', 'value': 'log'},
                      ],
                      value='linear',
                      labelStyle={'display': 'inline-block', 'margin': 'auto'},
                      id='radio-national-scale-selector', style={'width': '50%', 'display': 'inline-block',
                                                                 'align-items': 'center', 'justify-content': 'center',
                                                                 'margin-left': '30px', 'margin-right': '15px'}
                  )]),
        html.P([""" There is a weekly seasonal 
               pattern to the number of new cases reported which can be smoothed by taking a 7-day moving average. 
               Another thing to note is that by the end of August the number of recoveries a day had almost caught 
               up to the number of new cases before another wave of cases in September. The relative increase in confirmed cases, 
               recoveries and deaths can be better visualised by viewing the graphs on the log scale. Looking at the 
               relative recovery and death rate graph, the death rate appears to be much lower than the worldwide rate of 
               4% and has not grown at the same pace as the recovery rate. A spike can be seen in the recovery rate in mid-late
               May after the discharge policy was changed which allowed mild cases without fever to be discharged sans testing
               negative."""],
               style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
    ]),

    # Div - Testing Analysis
    html.Div(children=[
        html.H4(children='Testing Statistics', style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
                                                      'margin-right': '15px'}),
        dcc.Graph(
            id='testing-national'),

        html.Div([dcc.Dropdown(id='Testing-graph-drop',
                               options=[{'label': i, 'value': i} for i in
                                        ['Daily Testing and Cases', 'Moving Average',
                                         'Relative Metrics']],
                               value='Daily Testing and Cases', style={'width': '50%',
                                                                       'align-items': 'left',
                                                                       'justify-content': 'center',
                                                                       'margin-left': '10px', 'margin-right': '15px'}),
                  dcc.RadioItems(
                      options=[
                          {'label': 'Linear', 'value': 'linear'},
                          {'label': 'Log', 'value': 'log'},
                      ],
                      value='linear',
                      labelStyle={'display': 'inline-block', 'margin': 'auto'},
                      id='radio-national-test-scale', style={'width': '50%', 'display': 'inline-block',
                                                             'align-items': 'center', 'justify-content': 'center',
                                                             'margin-left': '30px', 'margin-right': '15px'}
                  )]),
        html.P(["""The Seasonality in the data seems to be due to a decrease in the collection of Testing Samples on
        the weekends. Another important thing to note is that ICMR releases the number of Testing Samples collected and 
        not the number of individuals tested. There is also a problem of data consistency as on certain days multiple bulletins are
        issued. Only the last numbers have been kept in such cases. A substantial increase in the number
         of Testing Samples collected can be seen from the middle of July. Looking at the relative metrics, the positive 
         rate can be used to determine whether enough tests are being done and has seen a decline since the increase in 
         testing. The tests per 1000 people is useful for tracking the penetration of the testing relative to the population.
        """], style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),

    ]),

    # Modelling state-wise statistics
    html.Div([html.H4(children='State-Wise Statistics ',
                      style={'textAlign': 'left',
                             'margin-top': '15px',
                             'margin-left': '15px',
                             'margin-right': '15px'}),
              dcc.Dropdown(
                  id='states-xaxis-column',
                  options=[{'label': i, 'value': i} for i in states],
                  value='Maharashtra'
              ),
              dcc.Graph(
                  id='state-stats'),
              html.P(["""Data and metrics for states; drop-down list sorted by the the total number of new cases in 
              the last week. The Growth Factor or daily Growth Rate is a metric used to track the exponential growth 
              of a pandemic and has been discussed later. Observe that some states like Delhi show clear evidence of 
              a second wave of cases."""]),
              ], style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),

    # Div - Modelling Cumulative Growth as a logistic function
    html.Div(children=[
        html.H4(children='Modeling Epidemic Growth : Logistic curve', style={'textAlign': 'left',
                                                                             'margin-top': '25px',
                                                                             'margin-left': '15px',
                                                                             'margin-right': '15px'}),
        html.P(["""The Logistic function is an example of a sigmoid curve and was devised as a model of population 
        growth. Epidemic growth is also observed to follow a logistic curve where the number of infected rises 
        exponentially and reaches a linear inflection point before gradually decreasing. Logistic functions have 
        applications in ecological modelling, medicine and especially in statistics and machine learning.""",
                """The logistic function is defined as :  \\begin{gather} f{(x)} = \\frac{L}{1 + e^{-k(x - x_0)}} \end{gather}
        $ \\text{Where, } x_0 = x\\text{ is the value of the sigmoids midpoint, } L =\\text{the curve's maximum value and, }$""",
                html.Br(),
                """ $ k =\\text{the logistic growth rate or steepness of the curve}$ """
                   , html.Br(), html.Br(), """A logistic curve can be observed in the case of China, but it is not 
                   reasonable to assume that all countries will follow similar trajectories, there may be multiple 
                   waves of exponential growth or the steepness of the curve may differ. Looking at the cumulative 
                   cases in India the next important tasks are determining whether the initial exponential growth has 
                   slowed down and whether the inflection point of the pandemic has been reached."""],
               style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
        dcc.Graph(
            id='sigmoid-china',
            figure=fig_china
        ),
        html.P([html.B("Fitting a Logistic curve to forecast cases"), html.Br(), """It is possible to fit a logistic 
        curve to the data and estimate the parameters to get a crude forecast for the curve of the virus. Using only 
        the 'days since first case' as a dependent variable and SciPy's optimise module we can use non-linear least 
        squares to fit a general logistic function and estimate the parameters :""", html.Br(), """L = 
        the maximum value of the curve or approximately or the total number of cumulative cases of of the virus.""",
                html.Br(),
                """x0 = the mid-point of the sigmoid or the approx date the inflection point of the virus growth is 
                reached.""", html.Br(), """Model performance is checked by using a validation set of the last 30 days 
                with $R^2$ and MAE as metrics. Since this is time-series data, it is not independent and thus was not 
                shuffled before being split into train-validation sets."""],
               style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify',
                      'margin-bottom': '5px'}),
        # Actual plot with the forecast
        dcc.Graph(
            id='fit-logistic', ),
        html.P(id='fit-logistic-stats',
               style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify',
                      'margin-bottom': '5px'}),
        html.Label(
            ['References : ', html.A('[1] 3Blue1Brown', href='https://www.youtube.com/watch?v=Kas0tIxDvrg'), html.Br(),
             html.A('[2] Journal of Medical Academics', href='https://www.maa.org/book/export/html/115630')],
            style={'margin-left': '25px', 'margin-right': '25px'}
        ),
    ]),

    # Modelling growth factor/rate of virus.
    html.Div(children=[
        html.H4(children='Growth Factor of Epidemic : Finding the Inflection Point', style={'textAlign': 'left',
                                                                                            'margin-top': '15px',
                                                                                            'margin-left': '15px',
                                                                                            'margin-right': '15px'}),
        html.P(["""The growth factor is a measure of exponential growth of an epidemic. The Growth Factor
        or growth rate on day N is defined as : \\begin{gather} G = \\frac{\Delta{C_{(N)}}}{\Delta{C_{(N-1)}}} \end{gather}
        I.e. the ratio of change in confirmed cases on day N and day N-1. The metric has a few helpful properties : """,
                html.Br(), """
        1. A value of greater than 1 signifies exponential growth.""", html.Br(), """ 
        2. A value of Less than 1 signifies decline.""", html.Br(), """3. A growth factor of 1 is the inflection 
        point and signifies the point where the growth of the epidemic is linear. """
                ],
               style={'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
        dcc.Graph(
            id='national-growth-factor'),
        dcc.Dropdown(
            id='radio-growth-factor-selector',
            options=[{'label': i, 'value': i} for i in ['Daily Growth Factor', 'Comparison of Growth Factor',
                                                        'Growth Factor Weekly Moving Average']],
            value='Daily Growth Factor', style={'width': '50%',
                                                'align-items': 'left', 'justify-content': 'center',
                                                'margin-left': '10px', 'margin-right': '15px'}
        ),

        html.P([""" As of the first week of September, the growth factor of the last month is much lower 
                   than the overall mean growth factor indicating that the initial exponential growth might have slowed down.
                   A weekly seasonal component can be seen in the growth factor and confirmed cases which is again due to the seasonal testing pattern.
                   Taking a weekly moving average to remove the effect of the seasonal component shows that the growth factor
                   still hasn't reached the inflection point and worryingly, the mean growth factor of the first week of 
                   September is higher than the mean growth factor of July indicating a second wave of cases.
                    """, html.Br(), html.B('Forecasting Growth Factor'), html.Br(),
                """Forecasting the Growth Factor is difficult to model as a regression problem due to a lack of 
                obvious features that can be engineered as predictors. Most models failed to outperform a simple mean 
                of the validation set and had negative $R^2$. For the regression models, lagged Growth Factor of the 
                past week along with date features are used. The Ridge Regression model with second degree polynomial 
                features and a tuned $l_2$ penalty was the only one that outperformed a simple forecast using the 
                mean of the training data. Traditional time-series models like a SARIMA(1, 1, 1)x(0, 1, 1, 
                7) performed much better. Growth Factor as a transformation is however destructive and cannot be used 
                to directly estimate the number of cases or cumulative cases but has been used as a feature to 
                another machine learning model later."""], style={'margin-top': '10px',
                                                                  'margin-left': '25px', 'margin-right': '25px',
                                                                  'textAlign': 'justify'}),
        dcc.Graph(id='forecast-growth-factor'),

    ]),

    ###### important for latex ######
    axis_latex_script,
    ###### important for latex ######
    mathjax_script,
])


@app.callback(
    [Output('national-stats', 'figure'),
     Output('testing-national', 'figure'),
     Output('national-growth-factor', 'figure'),
     Output('state-stats', 'figure'),
     Output('fit-logistic', 'figure'),
     Output('fit-logistic-stats', 'children'),
     Output('forecast-growth-factor', 'figure')],
    [Input('nat-graph-selector', 'value'),
     Input('radio-national-scale-selector', 'value'),
     Input('Testing-graph-drop', 'value'),
     Input('radio-national-test-scale', 'value'),
     Input('radio-growth-factor-selector', 'value'),
     Input('states-xaxis-column', 'value')],
)
def fetch_plots(value, value_scale, value_test, value_test_scale, value_gf, value_state):
    # national stats
    figure = india_national(india_data, value)
    if value_scale == 'log':
        figure.update_layout(yaxis_type="log")

    # Test stats
    figure_test = india_test_plot(india_test, value_test)
    if value_test_scale == 'log':
        figure_test.update_layout(yaxis_type="log")

    # growth factor
    figure_gf, min_val, max_val, start, stop = national_growth_factor(india_data, value_gf)
    # re-animate for growth factor comparison
    if value_gf == 'Comparison of Growth Factor':
        figure_gf.update_layout(yaxis_range=[min_val - 0.2, max_val + 0.2], xaxis_range=[start, stop])

    # state-wise analysis
    # if valid selection is missing, default to worst hit state
    if value_state is None:
        value_state = states[0]
    figure_state = state_plots(state_data[value_state], value_state)

    # forecast for logistic curve fit
    figure_log_curve, score_sigmoid_fit = forecast_curve_fit(india_data, x_data, y_data)
    # text with validation metrics for logistic curve
    log_fit_text = """The logistic curve seems to fit the general trend of the growth. For the validation set the """, \
                   html.B('R^2 is {:.4f}.'.format(score_sigmoid_fit['R^2'])), """ The """, \
                   html.B('Mean Absolute Error of {} '.format(int(score_sigmoid_fit['MAE']))), """ should show a less 
                   optimistic result as a simple logistic curve fit will not be able to account for seasonality or 
                   complex interactions. The results serves as a baseline for future models.""", html.Br(), """
                   Using the fit parameters of the function, it can be estimated that the Max Number of cases
                   or Peak of the curve will be {} cases. The Inflection point of the growth will be reached 
                   {} days after the 30th of Jan.""".format(int(score_sigmoid_fit['params']['L']),
                                                            int(score_sigmoid_fit['params']['x0']))

    # forecasts for growth factor
    figure_forecast_gf, score_gf_fit = forecast_growth_factor(india_data)

    return figure, figure_test, figure_gf, figure_state, figure_log_curve, log_fit_text, figure_forecast_gf


if __name__ == '__main__':
    app.run_server(host='localhost', port=8080, debug=True)
