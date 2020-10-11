# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_defer_js_import as dji
import pandas as pd
import numpy as np

# import functions and data from helper function

from utils.Dashboard_helper import india_national, log_epidemic_comp, country_df, india_test_plot, \
    national_growth_factor, make_state_dataframe, sharpest_inc_state, state_plots, forecast_curve_fit, \
    forecast_growth_factor, fetch_data, india_growth_ratio, forecast_growth_ratio, forecast_cases_growth_ratio

external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.18.1/styles/monokai-sublime.min.css']

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
Converted app layput to function to serve a dynamic layout on every page load.
"""


def serve_layout():
    return html.Div([
        # Overall header
        html.H1(children='Covid-19 : Data Analysis and Forecasting for India',
                style={'textAlign': 'center', 'margin-top': '20px'}),
        # header
        html.Div([html.P(['''Dashboard for analysing and forecasting the spread of the pandemic using Epidemiological,
    Time-Series and Machine Learning models and metrics.''',
                          html.Br(), html.B('Disclaimer : '),
                          '''The authors of the dashboard are not epidemiologists and any analysis and forecasting is 
                          not be taken seriously. The statistics, metrics and forecasts update daily and the analysis 
                          was performed in the first week of September.'''],
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
            by taking a 7-day moving average. Another thing to note is that by the end of August the number of 
            recoveries a day had almost caught up to the number of new cases before another wave of cases in 
            September. The relative increase in confirmed cases, recoveries and deaths can be better visualised by 
            viewing the graphs on the log scale. Looking at the relative recovery and death rate graph, 
            the death rate appears to be much lower than the worldwide rate of 4% and has not grown at the same pace 
            as the recovery rate. A spike is seen in the recovery rate in mid-late May after the discharge policy was 
            changed which allowed mild cases without fever to be discharged sans testing negative."""],
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
            on the weekends. Another important thing to note is that ICMR releases the number of Testing Samples 
            collected and not the number of individuals tested. There is also a problem of data consistency as on 
            certain days multiple bulletins are issued. Only the last numbers have been kept in such cases. A 
            substantial increase in the number of Testing Samples collected is seen from the middle of July. Looking 
            at the relative metrics, the positive rate can be used to determine whether enough tests are being done 
            and has seen a decline since the increase in testing. The tests per 1000 people is useful for tracking 
            the penetration of the testing relative to the population."""], style={'margin-top': '10px',
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
              the last week. The Growth Factor is a metric used to track the exponential growth 
              of a pandemic and has been discussed later. Observe that some states like Delhi show clear evidence of 
              a second wave of cases."""]),
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
            exponentially and reaches a linear inflection point before gradually decreasing. Logistic functions have 
            applications in ecological modelling, medicine and especially in statistics and machine learning.""",
                    """The logistic function is defined as : 
                     \\begin{gather} f{(x)} = \\frac{L}{1 + e^{-k(x - x_0)}} \end{gather} 
                     $ \\text{Where, } x_0 = x\\text{ is the value of the sigmoids midpoint, } L =\\text{the curve's maximum value and, }$""",
                    html.Br(), """ $ k =\\text{the logistic growth rate or steepness of the curve}$ """, html.Br(),
                    html.Br(), """A logistic curve is observed in the case of China, but it is not 
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
            html.P(["""The logistic curve seems to fit the general trend of the growth. For the validation set the 
            $ R^2 $ and Mean Absolute Error are shown in the table below. A simple logistic curve fit will not be able 
            to account for seasonality or complex interactions. An important thing to note is that $ R^2 $ is not the
             most suitable metric for time-series data and can give misleading results, more focus is on optimizing for 
             the MAE. The results here serve as a baseline for future models. Using the fit parameters of the function, 
            the Max Number of cases or Peak of the curve L and the Inflection point of the growth $x_0$ days since 30th 
            of Jan have also been estimated in the table below. Another thing to note is that the curve is fit on Cumulative
            Cases and the predictions have been differenced to show the predicted Daily Confirmed Cases. The evaluation 
             metrics have been calculated on these daily predictions and the transformation loses a data point."""],
                   style={'margin-top': '10px', 'margin-left': '25px', 'margin-right': '25px', 'textAlign': 'justify',
                          'margin-bottom': '5px'}),
            html.Div([dash_table.DataTable(
                id='Logistic-fit-table',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'R^2', 'id': 'R^2'},
                    {'name': 'MAE', 'id': 'MAE'},
                    {'name':  'Predicted Cumulative Cases', 'id': 'L'},
                    {'name': 'Predicted Date of Inflection Point', 'id': 'x0'}],
                data=[],
                sort_action="native",
                style_header={
                    'fontWeight': 'bold'
                }

            )], style={'margin-left': '25%', 'margin-right': '25%'}),
            html.P(
                ['References : ', html.A('[1] 3Blue1Brown', href='https://www.youtube.com/watch?v=Kas0tIxDvrg'),
                 html.Br(),
                 html.A('[2] Journal of Medical Academics', href='https://www.maa.org/book/export/html/115630')],
                style={'margin-left': '25px', 'margin-right': '25px'}
            ),
        ]),

        # Modelling growth factor/rate of virus.
        html.Div(children=[
            html.H4(children='Growth Factor of Epidemic : Finding the Inflection Point', style={'textAlign': 'left',
                                                                                                'margin-top': '15px',
                                                                                                'margin-left': '15px',
                                                                                                'margin-right': '15px'})
            , html.P(["""The growth factor is a measure of exponential growth of an epidemic. The Growth Factor
        on day n is defined as : \\begin{gather} Gf_{n} = \\frac{\Delta{C_{n}}}{\Delta{C_{n-1}}} \end{gather}
        I.e. the ratio of change in total or cumulative confirmed cases on day n and day n-1. This can also be framed 
        as the ratio of the number of new cases on day n and day n-1.
         The metric has a few helpful properties : """,
                      html.Br(), """
        1. A $ Gf > 1 $ signifies exponential growth in the number of new cases.""", html.Br(), """ 
        2. A $ Gf < 1 $ signifies decline or decay in the number of new cases.""", html.Br(), """3. A $ Gf = 1 $ is 
        the inflection point and signifies the point where exponential growth of the epidemic has stopped and is now 
        constant. """
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
                   A weekly seasonal component is seen in the growth factor and confirmed cases which is again due to the seasonal testing pattern.
                   Taking a weekly moving average to remove the effect of the seasonal component shows that the growth factor
                   still hasn't reached the inflection point and worryingly, the mean growth factor of the first week of 
                   September is higher than the mean growth factor of July indicating a second wave of cases.
                    """, html.Br(), html.H5('Forecasting Growth Factor'),
                    """Forecasting the Growth Factor is difficult to model as a regression problem due to a lack of 
                obvious features that can be engineered as predictors. For the regression models, lagged Growth Factor 
                of the past week along with date features are used. The Ridge Regression model with second degree 
                polynomial features and a tuned $l_2$ regularization was the only regression model that outperformed a 
                simple forecast using the mean of the training data. Traditional time-series models like SARIMA(1, 
                1, 1)x(0, 1, 1, 7) performed much better. Growth Factor as a transformation is destructive 
                and cannot be used to directly estimate the number of cases or cumulative cases. The growth factor forecast
                 has been used as a feature for other machine learning models later."""], style={'margin-top': '10px',
                                                                                                 'margin-left': '25px',
                                                                                                 'margin-right': '25px',
                                                                                                 'textAlign': 'justify'}),
            dcc.Graph(id='forecast-growth-factor'),

            html.Plaintext("""* Click on the legend and toggle visibility for one-one comparisons.""",
                           style={'margin-top': '0px',
                                  'margin-left': '25px', 'margin-right': '25px',
                                  'textAlign': 'right'}),
            html.Div([dash_table.DataTable(
                id='growth-factor-table',
                columns=[
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'R^2', 'id': 'R^2'},
                    {'name': 'MAE', 'id': 'MAE'},
                ],
                data=[],
                sort_action="native",
                style_header={
                    'fontWeight': 'bold'
                }

            )], style={'margin-left': '25%', 'margin-right': '25%'}),
        ]),

        # Modelling growth ratio of virus.

        html.Div([html.H4(children='Growth Ratio of Epidemic : A  Non-Destructive Metric', style={'textAlign': 'left',
                                                                                                  'margin-top': '15px',
                                                                                                  'margin-left': '15px',
                                                                                                  'margin-right': '15px'}),
                  html.P(["""The growth ratio is a metric that tracks the percentage Increase in total cumulative 
              cases from one day to the next. The growth ratio on day n is defined as :
               \\begin{gather} Gr_{n} = \\frac{{C_{n}}}{{C_{n-1}}} \end{gather}
               I.e. the ratio of total or cumulative confirmed cases on day n and 
              day n-1. The metric is not that interesting in itself but can be used to forecast the number of new 
              cases."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),
                  dcc.Graph(id="national-growth-ratio"),
                  html.P(["""For the growth ratio of cumulative cases, a decaying trend is expected and is observed. 
                  The ratio will tend to the lower bound of 1 when the number of new reported cases of the virus becomes 
                  negligible. The relative percentage growth in the number of cases has also gone down from a high of 
                  around 20% and is also much lower than the mean growth ratio. This could indicate that the initial 
                  exponential growth has most likely slowed down.
                  """, html.H5('Forecasting Growth Ratio Using Generalized Linear Models'), """
                  Looking at the decaying downwards trend, it could be possible to use Generalized Linear Models (GLM) 
                  with a suitable link function to model the problem. GLMs are used to model a relationship between predictors
                  $(X)$ and a non-normal target variable $(y)$. For a GLM with parameters $(w)$, the predicted values $(\hat{y})$
                  are linked to a linear combination of the input variables $(X)$ via an inverse link function $h$ as :
                  \\begin{gather} \hat{y}(w, X) = h(Xw). \end{gather}
                  So, the resulting model is still linear in parameters even though a non-linear relationship between 
                  the predictors and target is being modelled. GLMs have been introduced in Scikit-Learn recently (0.23).""",
                          html.Br(), html.B("Poisson Regression"), html.Br(), """Assumes the target to be from a 
                          Poisson distribution, typically used for data that is a count of occurrences of something 
                          in a fixed amount of time. The Poisson process is also commonly used to describe and model 
                          exponential decay.
                           The inverse link function $h$ used in Poisson regression is 
                         $h = \exp(Xw)$ and the target domain is $y \in [0, \infty)$. A value of 1 is subtracted from the 
                         growth ratio to get closer to meeting this constraint and should be fine for short-term forecasting."""
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
                  dcc.Graph(id="national-growth-ratio-preds"),
                  html.Div([dash_table.DataTable(
                      id='growth-ratio-table',
                      columns=[
                          {'name': 'Model', 'id': 'Model'},
                          {'name': 'R2', 'id': 'R2'},
                          {'name': 'MAE', 'id': 'MAE'},
                      ],
                      data=[],
                      sort_action="native",
                      style_header={
                          'fontWeight': 'bold'
                      }

                  )], style={'margin-left': '25%', 'margin-right': '25%'}),
                  html.Plaintext("""* Click on the legend and toggle visibility for other forecasts.""",
                                 style={'margin-top': '0px',
                                        'margin-left': '25px', 'margin-right': '25px',
                                        'textAlign': 'right'}),
                  html.P(["""Similar to the Growth Factor, an Autoregressive (AR(7)) model has been used as a 
                  baseline. Scikit-Learn applies $l_2$ regularisation to  estimators of the GLM class by default 
                  which resulted in poor performance and has not been used. Lagged growth ratio of the last 7-days 
                  along with date features have been used as features for the GLMs. For the GLMs, performing a 
                  polynomial transformation of degree 2 and scaling the input features led to better performance. The 
                  $R^2$ score is not a suitable metric for evaluating GLMs, but is still presented for consistency
                  in evaluating forecasts. The Gamma Regression model is the only regression model that is able 
                  to  outperform the baseline AR model.""", html.H5('Using Growth Ratio to Forecast Cases'),
                          """ The predicted Growth Ratio for day n can simply be multiplied with the Total cases of the
                           previous day to get the predicted Total confirmed cases on day n:
                            \\begin{gather} \hat{C_{n}} = C_{n-1}\\times{\hat{Gr_{n}}}\end{gather}
                            The predictions can then be differenced to get the predicted Daily cases on day n+1. This 
                            approach will lead to the loss of two data points during inference as the feature engineering
                            step also used the 1st difference of the 1st Lag as a feature."""],
                         style={'margin-top': '10px',
                                'margin-left': '25px',
                                'margin-right': '25px',
                                'textAlign': 'justify'}),

                  dcc.Graph(id="national-growth-ratio-cases-preds"),
                  html.Div([dash_table.DataTable(
                      id='growth-ratio-cases-table',
                      columns=[
                          {'name': 'Model', 'id': 'Model'},
                          {'name': 'R^2', 'id': 'R^2'},
                          {'name': 'MAE', 'id': 'MAE'},
                      ],
                      data=[],
                      sort_action="native",
                      style_header={
                          'fontWeight': 'bold'
                      }

                  )], style={'margin-left': '25%', 'margin-right': '25%'}),
                  ## TO DO

                  ## Finally Using Growth Ratio and Factor as features to final Ridge Regression model.
                  ## Compared to all models
                  ## Conclude that non-ML models can also perform well with relatively less tuning.
                  ## However with the limited available data/features. Better to stick to traditional models.

                  ]),

        ###### important for latex ######
        axis_latex_script,
        ###### important for latex ######
        mathjax_script,
    ])


# Define app layout to the actual function instance and not a call. This is for live-Updating
app.layout = serve_layout


@app.callback(
    [Output('national-stats', 'figure'),
     Output('testing-national', 'figure'),
     Output('national-growth-factor', 'figure'),
     Output('state-stats', 'figure'),
     Output('fit-logistic', 'figure'),
     Output('Logistic-fit-table', 'data'),
     Output('forecast-growth-factor', 'figure'),
     Output('growth-factor-table', 'data'),
     Output('national-growth-ratio', 'figure'),
     Output('national-growth-ratio-preds', 'figure'),
     Output('growth-ratio-table', 'data'),
     Output('national-growth-ratio-cases-preds', 'figure'),
     Output('growth-ratio-cases-table', 'data')],
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

    # Dict with validation metrics for logistic curve
    # time delta since start of series
    date_inflection = (india_data.index.min() + pd.Timedelta(days=int(score_sigmoid_fit['params']['x0']))).date().strftime('%d-%m-%Y')
    # required structure (List of dicts with each dict being a row) [{}, {}]
    log_fit_metrics = [{'Model': 'Logistic Curve Fit', 'R^2': round(score_sigmoid_fit['R^2'], 4),
                        'MAE': round(score_sigmoid_fit['MAE'], 4),
                        'L': int(score_sigmoid_fit['params']['L']), 'x0': date_inflection}]

    # forecasts for growth factor
    figure_forecast_gf, score_gf_fit, eval_metrics_gf = forecast_growth_factor(india_data)

    # plots for growth ratio

    fig_gr = india_growth_ratio(india_data)

    # forecasts for growth ratio

    figure_forecast_gr, eval_metrics_gr, preds_gr = forecast_growth_ratio(india_data)

    figure_forecast_cases_gr, eval_metrics_cases_gr = forecast_cases_growth_ratio(india_data, preds_gr)

    return figure, figure_test, figure_gf, figure_state, figure_log_curve, log_fit_metrics, figure_forecast_gf, \
           eval_metrics_gf, fig_gr, figure_forecast_gr, eval_metrics_gr, figure_forecast_cases_gr, eval_metrics_cases_gr


if __name__ == '__main__':
    app.run_server(host='localhost', port=8080, debug=True)
