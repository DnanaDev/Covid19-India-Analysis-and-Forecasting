# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_defer_js_import as dji
import pandas as pd

# import functions and data from helper function

from utils.Dashboard_helper import india_national, log_epidemic_comp, country_df, india_test_plot, \
    national_growth_factor, make_state_dataframe, sharpest_inc_state, state_plots, forecast_curve_fit

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

""" Data sources
"""

kaggle_snap_worldwide = pd.read_csv('utils/Data/kaggle_train.csv')
kaggle_snap_worldwide.rename(columns={'Country_Region': 'Country/Region',
                                      'Province_State': 'Province/State'}, inplace=True)
china_cases, china_fatal = country_df('China', kaggle_snap_worldwide)
fig_china = log_epidemic_comp(china_cases)

fig_growth_factor = national_growth_factor()

state_data = make_state_dataframe()

states = sharpest_inc_state(state_data)

""" App layout
"""
app.layout = html.Div([
    # Overall header
    html.H1(children='Covid-19 : India Analysis and Forecasting', style={'textAlign': 'center', 'margin-top': '20px'}),
    # header
    html.Div([html.P(['''An analysis of the current pandemic across the country. The first section can be 
    used as a dashboard to track the epidemic. The second section focuses on some mathematical and epidemiological metrics
     of the spread of the virus and attempts to forecast it.''', html.Br(), html.B('Note : '),
                      '''An attempt has been made to forecast the spread of the virus 
                  using traditional forecasting and machine learning models, however these results are to be taken with a grain of 
                  salt as the authors are not epidemiologists. The statistics and metrics update daily and the analysis was performed in the first week of 
                    September.'''],
                     style={'margin-left': '15px', 'margin-right': '15px'}),
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
                               value='Daily Statistics', style={'width': '50%', 'display': 'inline-block',
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
               up to the number of new cases before another wave of cases in September. The relative increase of confirmed cases, 
               recoveries, deaths can be better visualised by viewing the graphs on the log scale.""", html.Br(), """Looking at the 
               Relative recovery and death rate graph, the death rate appears to be much lower than the worldwide rate of 
               4% and has not grown at the same pace as the recovery rate. A spike can be seen in the recovery rate in mid-late
               May after the discharge policy was change."""],
               style={'textAlign': 'left', 'margin-top': '10px', 'margin-left': '15px',
                      'margin-right': '15px'}),
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
                               value='Daily Testing and Cases', style={'width': '50%', 'display': 'inline-block',
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
        issued. Only the last numbers have been kept in such cases.""", html.Br(), """A substantial increase in the number
         of Testing Samples collected can be seen from the middle of July. Looking at the relative metrics, the positive 
         rate can be used to determine whether enough tests are being done and has seen a decline since the increase in 
         testing. The tests per 1000 people is useful for tracking the penetration of the testing relative to the population.
        """], style={'textAlign': 'left', 'margin-top': '10px', 'margin-left': '15px',
                     'margin-right': '15px'}),

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
              html.P(["""Data and metrics for states, drop-down list sorted by the the total number of new cases in the
              last week. The Growth Factor or daily Growth rate is a metric for the exponential growth of a pandemic and 
              the details are discussed later.""", html.Br(), """Observe that some states like Delhi show clear evidence
              of a second wave of cases."""]),
              ], style={'textAlign': 'left', 'margin-top': '10px', 'margin-left': '15px',
                        'margin-right': '15px'}),

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
        $ \\text{Where, } x_0 = x\\text{ is the value of the sigmoids midpoint, } L =\\text{the curve's maximum value. and, } k =\\text{the logistic growth rate or steepness of the curve}$ """
                   , html.Br(), html.Br(), """A logistic curve can be observed in the case of China, but it is not 
                   reasonable to assume that all countries will follow similar trajectories, there may be multiple 
                   waves of exponential growth or the steepness of the curve may differ. Looking at the cumulative 
                   cases in India the next important tasks are determining whether the initial exponential growth has 
                   slowed down and whether the inflection point of the pandemic has been reached."""],
               style={'margin-left': '15px', 'margin-right': '15px'}),
        dcc.Graph(
            id='sigmoid-china',
            figure=fig_china
        ),
        html.P([html.B("Fitting a Logistic curve to forecast cases"), html.Br(), """It is possible to fit a logistic 
        curve to the data and estimate the parameters to get a crude forecast for the curve of the virus.""", html.Br(),
                """Using only the 'days since first case' as a dependent variable and SciPy's optimise module we can use 
        non-linear least squares to fit a general logistic function and estimate the parameters :""", html.Br(), """L = 
        the maximum value of the curve or approximately or the total number of cumulative cases of of the virus.""",
                html.Br(),
                """x0 = the mid-point of the sigmoid or the approx date the inflection point of the virus growth is 
                reached.""", html.Br(), """Model performance is checked by using a validation set of the last 30 days 
                with $R^2$ and MAE as metrics. Since, this is time-series data and the predictors are not 
                independent, the data was not shuffled before being split into train-validation sets."""],
               style={'margin-left': '15px', 'margin-right': '15px', 'margin-bottom': '5px'}),
        # Actual plot with the forecast
        dcc.Graph(
            id='fit-logistic', ),
        html.P(id='fit-logistic-stats',
               style={'margin-left': '15px', 'margin-right': '15px', 'margin-bottom': '5px'}),
        html.Label(
            ['References : ', html.A('[1] 3Blue1Brown', href='https://www.youtube.com/watch?v=Kas0tIxDvrg'), html.Br(),
             html.A('[2] Journal of Medical Academics', href='https://www.maa.org/book/export/html/115630')],
            style={'margin-left': '15px', 'margin-right': '15px'}
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
        2. A value of Less than 1 signifies decline.""", html.Br(), """
        3. A growth factor of 1 is the inflection point and signifies the point where the growth of the epidemic is no longer exponential."""
                ],
               style={'margin-left': '15px', 'margin-right': '15px'}),
        dcc.Graph(
            id='national-growth-factor'),
        html.Div([dcc.RadioItems(
            options=[
                {'label': 'Daily Growth Factor', 'value': 'daily'},
                {'label': 'Comparison of Growth Factor', 'value': 'comp'},
                {'label': 'Growth Factor Weekly Moving Average', 'value': 'moving'},
            ],
            value='daily',
            labelStyle={'display': 'inline-block', 'margin': 'auto'},
            id='radio-growth-factor-selector'
        )], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',
                   'margin-bottom': '15px'}),

        html.P([""" As of the first week of September, it can be seen that the growth factor of the last 30 days is much lower 
                   than the mean growth factor upto this point indicating that the initial exponential growth might have slowed down.
                   A weekly seasonal component can be seen in the growth factor and confirmed cases which can again be due to the seasonal testing pattern.
                   Taking a weekly moving average to remove the effect of the seasonal component shows that the growth factor
                   still hasn't reached the inflection point and worryingly there seems to be a second wave of growth as indicated 
                   by the growth factor of the previous week being higher than the growth factor of the last 30 days.
                    """, html.Br(), html.B('Forecasting Growth Factor'), html.Br(),
                """ The Growth Factor can be predicted using time-series forecasting methods like SARIMA
                    or other regression methods. The transformation is however destructive and cannot be used to directly 
                    estimate the number of cases or cumulative cases. 
                    """], style={'margin-left': '15px', 'margin-right': '15px'})

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
     Output('fit-logistic-stats', 'children')],
    [Input('nat-graph-selector', 'value'),
     Input('radio-national-scale-selector', 'value'),
     Input('Testing-graph-drop', 'value'),
     Input('radio-national-test-scale', 'value'),
     Input('radio-growth-factor-selector', 'value'),
     Input('states-xaxis-column', 'value')],
)
def fetch_plots(value, value_scale, value_test, value_test_scale, value_gf, value_state):
    # national stats
    figure = india_national(value)
    if value_scale == 'log':
        figure.update_layout(yaxis_type="log")

    # Test stats
    figure_test = india_test_plot(value_test)
    if value_test_scale == 'log':
        figure_test.update_layout(yaxis_type="log")

    # growth factor
    figure_gf, min_val, max_val, start, stop = national_growth_factor(value_gf)
    # re-animate for growth factor comparison
    if value_gf == 'comp':
        figure_gf.update_layout(yaxis_range=[min_val - 0.2, max_val + 0.2], xaxis_range=[start, stop])

    # state-wise analysis
    # if valid selection is missing, default to worst hit state
    if value_state is None:
        value_state = states[0]
    figure_state = state_plots(state_data[value_state], value_state)

    # forecast for logistic curve fit
    figure_log_curve, score_sigmoid_fit = forecast_curve_fit()
    # text with validation metrics for logistic curve
    log_fit_text = """The logistic curve seems to fit the general trend of the growth. For the validation set the """, \
                   html.B('R^2 is {:.4f}.'.format(score_sigmoid_fit['R^2'])), """ The """, \
                   html.B('Mean Absolute Error of {} '.format(int(score_sigmoid_fit['MAE']))), """ should show a less 
                   optimistic result as a simple logistic curve fit will not be able to account for seasonality or 
                   complex interactions. The results serves as a baseline for future models.""", html.Br(), """
                   Using the fit parameters of the function, it can be estimated that the Max Number of cases
                   or Peak of the curve will be {} cases. The Inflection point of the growth will be reached at 
                   {} days since the 1st of Jan.""".format(int(score_sigmoid_fit['params']['L']),
                                                           int(score_sigmoid_fit['params']['x0']))
    return figure, figure_test, figure_gf, figure_state, figure_log_curve, log_fit_text


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
