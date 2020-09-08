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
    national_growth_factor, make_state_dataframe, sharpest_inc_state, state_plots

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
     of the spread of the virus.''', html.Br(), html.B('Note : '),
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
                                                                           'Relative Growth Ratio, Recovery, Death rate']],
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
        html.P(["""As of the first week of September the number of Daily confirmed cases has reached almost 100K cases a 
               day and the total number of cases has crossed 4 million.""", html.Br(), """ There is a weekly seasonal 
               pattern to the number of new cases reported which can be smoothed by taking a 7-day moving average. 
               Another thing to note is that by the end of August the number of recoveries a day had almost caught 
               up to the number of new cases before another wave of cases in September.""", html.Br(), """Looking at the 
               third graph, the death rate also appears to be much lower than the worldwide rate of 4% and 
               has not grown at the pace of the new reported cases or the recovery rate."""],
               style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
                      'margin-right': '15px'}),
    ]),

    # Div - Testing Analysis
    html.Div(children=[
        html.H4(children='Testing Trends', style={'textAlign': 'left', 'margin-top': '25px', 'margin-left': '15px',
                                                  'margin-right': '15px'}),
        dcc.Graph(
            id='testing-national'),

        html.Div([dcc.RadioItems(
            options=[
                {'label': 'Daily Testing', 'value': 'daily_test'},
                {'label': 'Daily Testing vs Daily Cases', 'value': 'daily_test_case'},
                {'label': 'Daily Testing vs Daily Cases(Log)', 'value': 'daily_test_case_log'},
            ],
            value='daily_test',
            labelStyle={'display': 'inline-block', 'margin': 'auto'},
            id='radio-national-test-selector'
        )
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        html.P("""The comparison of the graphs of the amount of Testing Done and the confirmed Cases each day can be a 
        good indicator of whether enough testing is being done. If the curves are very close together it can indicate 
        that not enough people are getting tested. Visualising the Daily confirmed cases and Testing Samples collected 
        on a semi-log graph shows that the number of confirmed cases per day is just in pace with the Testing Rate. 
        Also judging by the sharp increase in samples collected around April, we can say that testing is finally being
         done on a large enough scale. Also observe the weekly seasonality in testing samples 
        collected.""", style={'margin-left': '15px', 'margin-right': '15px'}),

    ]),

    # Modelling state-wise statistics
    html.Div([html.H4(children='State-Wise Statistics : Which states are handling the Virus well ?',
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
              ]),

    # Div - Modelling Cumulative Growth as a logistic function
    html.Div(children=[
        html.H4(children='Modeling Epidemic Growth : Logistic or Sigmoid curve', style={'textAlign': 'left',
                                                                                        'margin-top': '25px',
                                                                                        'margin-left': '15px',
                                                                                        'margin-right': '15px'}),
        html.P([""" Epidemics are observed to follow a logistic curve. The no. of infected rises 
            exponentially and reaches an inflection point before gradually decreasing. Logistic functions have applications in
            ecological modelling, medicine and especially in statistics and machine learning. The logistic function is 
                defined as :  \\begin{gather} f{(x)} = \\frac{L}{1 + e^{-k(x - x_0)}} \end{gather} $ \\text{Where, } x_0 = x\\text{ is the value of the sigmoids midpoint, } L =\\text{the curve's maximum value. and, } k =\\text{the logistic growth rate or steepness of the curve}$"""
                   , html.Br(), """ This can be seen somewhat in the case of China as shown in the figure below, but it is not reasonable to 
                assume that all countries will follow a similar trajectory, there may be multiple waves of exponential growth
                 or the steepness of the curve may differ. However, comparing to the cases 
                in India it can be seen that the initial exponential growth seems to have slowed down. The next important 
                milestone is whether the inflection point has been reached.
                """], style={'margin-left': '15px', 'margin-right': '15px'}),
        dcc.Graph(
            id='sigmoid-china',
            figure=fig_china
        ),
        html.Label(
            ['References : ', html.A('[1] 3Blue1Brown', href='https://www.youtube.com/watch?v=Kas0tIxDvrg'), html.Br(),
             html.A('[2] Journal of Medical Academics', href='https://www.maa.org/book/export/html/115630')],
            style={'margin-left': '15px', 'margin-right': '15px'}
        )
    ]),

    # Modelling growth factor/rate of virus.
    html.Div(children=[
        html.H4(children='Growth Factor Of Epidemic : Finding the Inflection Point', style={'textAlign': 'left',
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
                   , html.Br(), html.Br(), """ As of the end of August, it can be seen that the growth factor of the last 30 days is much lower 
                   than the mean growth factor upto this point indicating that the initial exponential growth might have slowed down.
                   A weekly seasonal component can be seen in the growth factor and confirmed cases which can again be due to the seasonal testing pattern.
                   Taking a weekly moving average to remove the effect of the seasonal component shows that the growth factor
                   still hasn't reached the inflection point and worryingly there seems to be a second wave of growth as indicated 
                   by the growth factor of the previous week being higher than the growth factor of the last 30 days.
                    """],
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
        )], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

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
     Output('state-stats', 'figure')],
    [Input('nat-graph-selector', 'value'),
     Input('radio-national-scale-selector', 'value'),
     Input('radio-national-test-selector', 'value'),
     Input('radio-growth-factor-selector', 'value'),
     Input('states-xaxis-column', 'value')],
)
def fetch_national_plot(value, value_scale, value_test, value_gf, value_state):
    # national stats
    figure = india_national(value)
    if value_scale == 'log':
        figure.update_layout(yaxis_type="log")
    # Test stats
    figure_test = india_test_plot(value_test)

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
    return figure, figure_test, figure_gf, figure_state


if __name__ == '__main__':
    app.run_server(debug=True)
