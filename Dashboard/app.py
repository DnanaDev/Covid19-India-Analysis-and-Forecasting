# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_defer_js_import as dji
import pandas as pd

# import functions and data from helper function

from utils.Dashboard_helper import india_national, country_df, india_test_plot, \
    national_growth_factor, make_state_dataframe, sharpest_inc_state, state_plots, static_forecast_plots, fetch_data, \
    forecast_cases

external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css',
                        dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# for gunicorn server
server = app.server

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

### wrangling Data/ Data for forecasting models ###
china_cases, china_fatal = country_df('China', kaggle_snap_worldwide)

states = sharpest_inc_state(state_data)

india_data_combined = india_data.join(india_test_series, how='left')

# making copy for test data functions
india_test = india_data_combined.copy()

### Static Figures and Forecasts ###

fig_growth_factor = national_growth_factor(india_data)

figure_log_curve, log_fit_metrics, figure_forecast_gf, eval_metrics_gf, fig_gr, figure_forecast_gr, \
eval_metrics_gr, figure_forecast_cases_gr, eval_metrics_cases_gr = static_forecast_plots(india_data)

"""
Converted app layout to function to serve a dynamic layout on every page load.
"""


def serve_layout():
    return html.Div([
        # Overall header
        html.H1(children='Covid-19 : Data Analysis and Forecasting for India',
                style={'textAlign': 'center', 'margin-top': '20px'}),
        # header
        html.Div([html.P(['''Dashboard for analysing and forecasting the spread of the pandemic using Mathematical 
        curve fitting, Time-Series forecasting and Supervised Machine Learning models and metrics. The metrics and 
        forecasts update on app load and any subjective analysis was written in the first week of September.''',
                          html.Br(), html.B('Disclaimer : '),
                          '''The authors of the dashboard are not epidemiologists and any analysis and forecasting is 
                          not be taken seriously.'''
                          ],
                         style={'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
                  ]),

        # Div - Nationwide Analysis
        html.Div([
            html.H4('Nation-wide Statistics', style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
                                                     'margin-right': '15px'}),
            dcc.Graph(
                id='national-stats'),
            html.Div([dcc.Dropdown(id='nat-graph-selector',
                                   options=[{'label': i, 'value': i} for i in
                                            ['Daily Statistics', 'Cumulative Statistics',
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
                                                                     'align-items': 'center',
                                                                     'justify-content': 'center',
                                                                     'margin-left': '30px', 'margin-right': '15px'}
                      )]),
            html.P(["""There is a weekly seasonal pattern to the number of new cases reported which can be smoothed 
            by taking a 7-day moving average. The relative increase in confirmed cases, recoveries and deaths can be 
            better visualised by viewing the graphs on the log scale. Looking at the relative recovery and death rate 
            graph, the death rate appears to be much lower than the worldwide rate of 4%."""],
                   style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
        ]),

        # Div - Testing Analysis
        html.Div(children=[
            html.H4(children='Testing Statistics',
                    style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
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
                                                                           'margin-left': '10px',
                                                                           'margin-right': '15px'}),
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
            html.P(["""The Seasonality in the data seems to be due to a decrease in the collection of Testing Samples 
            on the weekends. The positivity rate tracks the percentage of tests that results in a positive and 
            is used to determine whether enough tests are being done. 
            The tests per 1000 people is useful for tracking the penetration of the testing relative to the 
            population size."""], style={'margin-top': '10px',
                                         'margin-left': '25px',
                                         'margin-right': '25px',
                                         'textAlign': 'justify'}),

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
              the state for the last week. The Growth Factor is a metric used to track the exponential growth 
              of a pandemic and has been discussed later. Observe that some states like Delhi show clear evidence of 
              multiple waves of cases."""]),
                  ],
                 style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),

        # Div - Modelling Cumulative Growth as a logistic function
        html.Div(children=[
            html.H4(children='Modeling Epidemic Growth : Logistic curve', style={'textAlign': 'left',
                                                                                 'margin-top': '25px',
                                                                                 'margin-left': '15px',
                                                                                 'margin-right': '15px'}),
            html.P(["""The Logistic function is an example of a sigmoid curve and was devised as a model of population 
            growth. Epidemic growth is also observed to follow a logistic curve where the number of infected rises 
            exponentially and reaches an inflection point before gradually decreasing.""",
                    """The logistic function is defined as : 
                     \\begin{gather} f{(x)} = \\frac{L}{1 + e^{-k(x - x_0)}} \end{gather} 
                     $ \\text{Where, } x_0 = x\\text{ is the value of the sigmoids midpoint, }$""",
                    html.Br(), """ $ L =\\text{the curve's maximum value and, }$""",
                    html.Br(), """ $ k =\\text{the logistic growth rate or steepness of the curve}$ """, html.Br(),
                    html.Br(), """The presence of 
                    multiple waves of exponential growth can be a problem for such a simple model and is just used as a 
                    starting point. For confirmed cases in India an important task is to determine whether the initial 
                    exponential growth has slowed down and whether the inflection point of the pandemic has been reached."""],
                   style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
            html.P([html.H5("Fitting a Logistic curve to forecast cases"), """Using only 
        the 'days since first case' as a dependent variable and SciPy's optimise module we use non-linear least 
        squares to fit a general logistic function and estimate the parameters :""", html.Br(), """L = 
        Predicted total number of cumulative cases of of the virus.""",
                    html.Br(),
                    """x0 = Predicted date the inflection point of the virus growth is 
                reached."""],
                   style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify',
                          'margin-bottom': '5px'}),
            # Actual plot with the forecast
            dcc.Graph(
                id='fit-logistic',
                figure=figure_log_curve),
            html.P(["""A simple logistic curve fit will not be able to account for seasonality or complex 
            interactions. Since this is time-series data, 
            the samples are not independent and the data was not shuffled before being split into train-validation 
            sets. 
             The evaluation metrics have been 
            calculated on the daily predictions and the transformation from total to daily cases loses a data point."""],
                   style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify',
                          'margin-bottom': '15px'}),
            html.Div([dash_table.DataTable(
                id='Logistic-fit-table',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'R^2', 'id': 'R^2'},
                    {'name': 'MAPE(%)', 'id': 'MAPE'},
                    {'name': 'RMSE(Cases)', 'id': 'RMSE'},
                    {'name': 'Predicted Cumulative Cases (L)', 'id': 'L'},
                    {'name': 'Predicted Date of Inflection Point ($x_0$)', 'id': 'x0'}],
                data=log_fit_metrics,
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                sort_action="native",
                style_header={
                    'fontWeight': 'bold'
                }
            )], style={'margin-left': '25%', 'margin-right': '25%'}),
            html.Div(
                [html.P([html.B('Evaluation Metrics reported:'), html.Br(), """1. The R-Squared ($R^2$) score 
                is not particularly useful for forecast quality evaluation but is useful for comparing to the null model 
                ( which always predicts the mean of the data) and across models.""", html.Br(), """2. The Mean Absolute 
                    Percentage Error (MAPE) is a simple metric that can be interpreted as a percentage and favors 
                    forecasts that underestimate cases rather than ones that will overestimate it. It is unbounded for 
                    large positive errors.""",
                         html.Br(), """3. The Root Mean Squared Error (RMSE) penalises larger errors more severely and
                    can be interpreted in the target or 'cases' scale.
            """], style={'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'})])
        ]),

        # Modelling growth factor/rate of virus.
        html.Div(children=[
            html.H4(children='Growth Factor of Epidemic : Finding the Inflection Point', style={'textAlign': 'left',
                                                                                                'margin-top': '15px',
                                                                                                'margin-left': '15px',
                                                                                                'margin-right': '15px'})
            , html.P(["""The growth factor is the factor by which a quantity multiplies itself over time, in our case
            the growth factor would measure the factor by which the daily new cases grow. It can be used to find insights 
            into the exponential growth of an epidemic. The Growth Factor on day n is defined as :
            \\begin{gather} Gf_{n} = \\frac{\Delta{C_{n}}}{\Delta{C_{n-1}}} \end{gather}
        I.e. the ratio of change in total or cumulative confirmed cases on day n and day n-1. This can also be framed 
        as the ratio of the number of new cases on day n and day n-1.
         The metric has a few helpful properties : """,
                      html.Br(), """
        1. A $ Gf > 1 $ signifies exponential growth in the number of new cases.""", html.Br(), """ 
        2. A $ Gf < 1 $ signifies decline or decay in the number of new cases.""", html.Br(), """3. A $ Gf = 1 $ is 
        the inflection point and signifies the point where exponential growth of the epidemic has plateaued."""
                      ],
                     style={'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify'}),
            dcc.Graph(
                id='national-growth-factor'),
            dcc.Dropdown(
                id='radio-growth-factor-selector',
                options=[{'label': i, 'value': i} for i in ['Growth Factor Weekly Moving Average',
                                                            'Comparison of Growth Factor', 'Daily Growth Factor']],
                value='Growth Factor Weekly Moving Average', style={'width': '50%',
                                                                    'align-items': 'left', 'justify-content': 'center',
                                                                    'margin-left': '10px', 'margin-right': '15px'}
            ),

            html.P([""" If the current growth factor is much lower than median growth factor
            and also consistently below the inflection point, it might indicate that the initial exponential growth of 
            cases is over. The comparison of growth factor graph can also be useful for identifying subsequent waves of 
            cases by comparing the mean growth factor of the last week, month and overall mean growth factor.
            
                    """, html.Br(), html.H5('Forecasting Growth Factor'),
                    """A SARIMA(1, 1, 1)x(0, 1, 1, 7) model that takes into account the weekly seasonality is used as 
                    a traditional time-series forecasting baseline. The mean growth factor of the last month before 
                    validation split is also provided as a baseline.""", html.Br(), html.B('Forecasting as a '
                                                                                           'Supervised Learning '
                                                                                           'Problem'), html.Br(),
                    """To use traditional supervised learning algorithms with time-series, features are
                    extracted that take into account the time aspect of the data, such as the  """,
                    html.B('lag'), """ of the Growth Factor for the last 7 days and """, html.B('date features'), """ (
                    day of the month, day of the week). A Linear Regression 
                    and a Ridge Regression model are used here to forecast the growth factor. The Ridge Regression 
                    model is used with second-degree polynomial features and $l_2$ regularization.""", html.Br(),
                    html.B('Recursive Multi-step Forecasting'), html.Br(), """The regression models use multiple lags 
                     of the target (t-1 to t-n) as input features. This is a problem at inference and validation time 
                     as predictions at t+2 would need lags of the target that aren't available. A solution is to 
                     recursively predict the growth factor for the next data and use it as a lag feature on the next day.
                     The validation performance has been evaluated with this recursive approach."""],
                   style={'margin-top': '10px',
                          'margin-left': '25px',
                          'margin-right': '25px',
                          'textAlign': 'justify'}),
            dcc.Graph(id='forecast-growth-factor',
                      figure=figure_forecast_gf),
            html.Div([dash_table.DataTable(
                id='growth-factor-table',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'R^2', 'id': 'R^2'},
                    {'name': 'MAPE(%)', 'id': 'MAPE'},
                    {'name': 'RMSE', 'id': 'RMSE'},
                ],
                data=eval_metrics_gf,
                sort_action="native",
                style_header={
                    'fontWeight': 'bold'
                }

            )], style={'margin-left': '25%', 'margin-right': '25%'}),
            html.P([""" Growth Factor as a transformation is destructive and cannot be used to directly estimate the 
            number of cases or forecast the cumulative cases. It is still an important metric for understand the pattern 
            of the growth in the number of cases."""],
                   style={'margin-top': '10px',
                          'margin-left': '25px',
                          'margin-right': '25px',
                          'textAlign': 'justify'})
        ]),

        # Forecasting Daily Cases.
        html.Div(children=[html.H4('Forecasting Daily New Cases', style={'textAlign': 'left',
                                                                         'margin-top': '15px',
                                                                         'margin-left': '15px',
                                                                         'margin-right': '15px'}),
                           html.P(["""A SARIMA(1, 1, 0)x(0, 1, 1, 7) is used as a traditional time-series forecasting 
                           baseline. For the supervised learning approach, A Lasso regression model with sliders for 
                           feature extraction and hyper-parameter tuning is provided below. A Lasso regression model 
                           applies $l_1$ regularisation to create sparse models that automatically perform feature 
                           selection. Date features in addition to the lagged target features are used and a recursive
                           approach is used for the forecast when using lag features."""],
                                  style={'margin-top': '10px',
                                         'margin-left': '25px',
                                         'margin-right': '25px',
                                         'textAlign': 'justify'}),
                           ], style={'padding': '5px'}
                 ),
        dbc.Container(
            fluid=True,
            children=[
                dbc.Row([
                    dbc.Col([dbc.Label('Features and Hyper-parameters:', size='md', style={'margin-top': 20}),
                             html.Div([dbc.Label('Target Lags:'),
                                       dcc.Slider(id='cases-lag-sel',
                                                  min=1,
                                                  max=21,
                                                  value=7,
                                                  marks={
                                                      7: {'label': '7-Lags',
                                                          'style': {
                                                              'color': '#77b0b1'}},
                                                      14: {'label': '14-Lags'},
                                                      21: {
                                                          'label': '21-Lags'}, },
                                                  )], style={'margin-top': 20}),
                             html.Div([dbc.Label('L1 Regularisation:'),
                                       dcc.Slider(id='cases-regularisation-sel',
                                                  min=1,
                                                  max=9000,
                                                  value=4500,
                                                  marks={
                                                      1: {'label': '1 (Default)',
                                                          'style': {'color': '#77b0b1'}},
                                                      4500: {'label': '4500'},
                                                      9000: {'label': '9000'},
                                                  })], style={'margin-top': 20}),
                             html.Div([dbc.Label('Polynomial Features:'),
                                       dcc.Slider(
                                           id='cases-poly-sel',
                                           min=1,
                                           max=3,
                                           value=1,
                                           marks={
                                               1: {'label': '1 (None)',
                                                   'style': {'color': '#77b0b1'}},
                                               2: {'label': '2nd Degree'},
                                               3: {'label': '3rd Degree'},
                                           })], style={'margin-top': 20}),
                             ], width=2),
                    dbc.Col([dcc.Graph(id="national-pred-cases")], width=6),
                    dbc.Col([dcc.Graph(id="national-pred-feat-imp")], width=4)
                ]), html.Plaintext("""* Not all features shown in figure.""",
                                   style={'margin-top': '0px',
                                          'margin-left': '25px', 'margin-right': '25px',
                                          'textAlign': 'right'}),
            ]),

        html.Div([dash_table.DataTable(
            id='Cases-fit-table',
            columns=[
                {'name': 'Model', 'id': 'Model'},
                {'name': 'R^2', 'id': 'R^2'},
                {'name': 'MAPE(%)', 'id': 'MAPE'},
                {'name': 'RMSE(Cases)', 'id': 'RMSE'}],
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            sort_action="native",
            style_header={
                'fontWeight': 'bold'
            }
        )], style={'margin-left': '25%', 'margin-right': '25%'}),

        # Modelling growth ratio of virus.
        html.Div([html.H4(children='Growth Ratio of Epidemic : A  Non-Destructive Transformation',
                          style={'textAlign': 'left',
                                 'margin-top': '15px',
                                 'margin-left': '15px',
                                 'margin-right': '15px'}),
                  html.P(["""The growth ratio is a metric that represents the percentage Increase in total cumulative 
              cases from one day to the next. The growth ratio on day n is defined as :
               \\begin{gather} Gr_{n} = \\frac{{C_{n}}}{{C_{n-1}}} \end{gather}
               I.e. the ratio of total or cumulative confirmed cases on day n and 
              day n-1. The transformation is not that interesting in itself but can be used to forecast the number of new 
              cases."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),
                  dcc.Graph(id="national-growth-ratio",
                            figure=fig_gr),
                  html.P(["""For the growth ratio of cumulative cases, a decaying trend is observed.
                  As the number of reported new cases of the virus becomes negligible the ratio of cumulative 
                  cases will tend to its lower bound of 1.""",
                          html.H5('Forecasting Growth Ratio Using Generalized Linear Models'), """
                  Looking at the decaying trend, it could be useful to fit a Generalized Linear Models (GLM) 
                  with a suitable link function to the data. GLMs are used to model a relationship between predictors
                  $(X)$ and a non-normal target variable $(y)$. For a GLM with parameters $(w)$, the predicted values $(\hat{y})$
                  are linked to a linear combination of the input variables $(X)$ via an inverse link function $h$ as :
                  \\begin{gather} \hat{y}(w, X) = h(Xw). \end{gather}
                  So, the resulting model is still linear in parameters even though a non-linear relationship between 
                  the predictors and target is being modelled.""",
                          html.Br(), html.B("Poisson Regression"), html.Br(), """Assumes the target to be from a 
                          Poisson distribution, typically used for data that is a count of occurrences of something 
                          in a fixed amount of time. The Poisson process is also commonly used to describe and model 
                          exponential decay.
                           The inverse link function $h$ used in Poisson regression is 
                         $h = \exp(Xw)$ and the target domain is $y \in [0, \infty)$. A value of 1 is subtracted from the 
                         growth ratio almost meet this constraint."""
                             , html.Br(), html.B("Gamma Regression"), html.Br(), """
                          Assumes the target to be from a 
                          Gamma distribution, which is typically used to model exponential target data. 
                          The inverse link function $h$ used in Gamma regression is 
                         $h = -(Xw)^{-1}$ and the target domain is $y \in (0, \infty)$. Shifting the growth ratio by 1 meets 
                         this constraint."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),
                  html.P(["""A traditional forecasting Autoregressive AR(7) model 
                  has been used as a baseline. No regularisation has been used. Degree 2 polynomial transformed and 
                  scaled Lag of growth ratio for the previous 7-days 
                  along with date features has been used as features for the GLMs. The forecasts
                  are performed and evaluated with a recursive multi-step approach. As the train-test split was done
                  before creating lag features the first 7 data points have been dropped."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),
                  dcc.Graph(id="national-growth-ratio-preds",
                            figure=figure_forecast_gr),
                  html.Div([dash_table.DataTable(
                      id='growth-ratio-table',
                      columns=[
                          {'name': 'Model', 'id': 'Model'},
                          {'name': 'R2', 'id': 'R2'},
                          {'name': 'MAPE(%)', 'id': 'MAPE'},
                          {'name': 'RMSE(GR)', 'id': 'RMSE'},
                      ],
                      data=eval_metrics_gr,
                      sort_action="native",
                      style_header={
                          'fontWeight': 'bold'
                      }

                  )], style={'margin-left': '25%', 'margin-right': '25%'}),
                  html.Plaintext("""* Click on the legend and toggle visibility for other forecasts.""",
                                 style={'margin-top': '0px',
                                        'margin-left': '25px', 'margin-right': '25px',
                                        'textAlign': 'right'}),
                  html.P([html.H5('Using Growth Ratio to Forecast Cases'),
                          """ The predicted Growth Ratio for day n can simply be multiplied with the Total cases of the
                           previous day to get the predicted Total confirmed cases on day n:
                            \\begin{gather} \hat{C_{n}} = C_{n-1}\\times{\hat{Gr_{n}}}\end{gather}
                            The predictions can then be differenced to get the predicted Daily cases on day n+1. This 
                            approach will lead to the loss of a couple of data points at validation time."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),

                  dcc.Graph(id="national-growth-ratio-cases-preds",
                            figure=figure_forecast_cases_gr),
                  html.Div([dash_table.DataTable(
                      id='growth-ratio-cases-table',
                      columns=[
                          {'name': 'Model', 'id': 'Model'},
                          {'name': 'R^2', 'id': 'R^2'},
                          {'name': 'MAPE(%)', 'id': 'MAPE'},
                          {'name': 'RMSE(GR)', 'id': 'RMSE'}
                      ],
                      data=eval_metrics_cases_gr,
                      sort_action="native",
                      style_header={
                          'fontWeight': 'bold'
                      }

                  )], style={'margin-left': '25%', 'margin-right': '25%'}),
                  html.Div([
                      html.P([html.H5('Insights and Summary'), """Multiple different metrics for tracking the spread of 
                      the Coronavirus pandemic are presented here. Looking at the performance of the different forecasting
                      methods it is evident that traditional forecasting models prove to be good baseline models in data-starved
                      situations. Supervised machine learning models can outperform the traditional
                       models after careful feature extraction and tuning but generally provide less stable results."""]
                             )], style={'margin-top': '10px',
                                        'margin-left': '25px',
                                        'margin-right': '25px',
                                        'textAlign': 'justify'}),
                  html.Div([html.P([html.B('References and Data Sources: '),
                                    html.Br(),
                                    html.A('[1] 3Blue1Brown', href='https://www.youtube.com/watch?v=Kas0tIxDvrg'),
                                    html.Br(),
                                    html.A('[2] Journal of Medical Academics',
                                           href='https://www.maa.org/book/export/html/115630'),
                                    html.Br(), html.A('[3] Daily Case Statistics: covid19india.org API',
                                                      href='https://github.com/covid19india/api'),
                                    html.Br(),
                                    html.A('[4] Daily ICMR Testing Data: Data Meet',
                                           href='https://github.com/datameet/covid19')

                                    ])], style={'margin-left': '25px', 'margin-right': '25px'})
                  ]),

        ###### important for latex ######
        axis_latex_script,
        ###### important for latex ######
        mathjax_script,
    ])


# Define app layout to the actual function instance and not a call. This is for live-Updating
app.layout = serve_layout


@app.callback([
    Output('national-stats', 'figure'),
    Output('testing-national', 'figure'),
    Output('state-stats', 'figure')],
    [Input('nat-graph-selector', 'value'),
     Input('radio-national-scale-selector', 'value'),
     Input('Testing-graph-drop', 'value'),
     Input('radio-national-test-scale', 'value'),
     Input('states-xaxis-column', 'value')])
def fetch_dashboard_plots(value, value_scale, value_test, value_test_scale, value_state, ):
    # national stats
    figure = india_national(india_data, value)
    if value_scale == 'log':
        figure.update_layout(yaxis_type="log")

    # Test stats
    figure_test = india_test_plot(india_test, value_test)
    if value_test_scale == 'log':
        figure_test.update_layout(yaxis_type="log")
    # state-wise analysis
    # if valid selection is missing, default to worst hit state
    if value_state is None:
        value_state = states[0]
    figure_state = state_plots(state_data[value_state], value_state)

    return figure, figure_test, figure_state,


@app.callback([
    Output('national-growth-factor', 'figure'),
    Output('national-pred-cases', 'figure'),
    Output('Cases-fit-table', 'data'),
    Output('national-pred-feat-imp', 'figure')], [
    Input('radio-growth-factor-selector', 'value'),
    Input('cases-lag-sel', 'value'),
    Input('cases-regularisation-sel', 'value'),
    Input('cases-poly-sel', 'value')])
def fetch_dynamic_forecast_plots(value_gf, num_lags, alpha, poly):
    # growth factor
    figure_gf, min_val, max_val, start, stop = national_growth_factor(india_data, value_gf)
    # re-animate for growth factor comparison
    if value_gf == 'Comparison of Growth Factor':
        figure_gf.update_layout(yaxis_range=[min_val - 0.2, max_val + 0.2], xaxis_range=[start, stop])

    # forecast cases
    figure_case_preds, case_preds_eval, figure_feat_imp = forecast_cases(india_data.DailyConfirmed[41:],
                                                                         num_lags=int(num_lags), alpha=alpha,
                                                                         poly_feats=int(poly))
    return figure_gf, figure_case_preds, case_preds_eval, figure_feat_imp


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False)
