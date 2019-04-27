import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    # Header
    html.Div([
        html.H2(
            'Exercise Boundary of American Option',
            style={'float': 'center',}
        ),   
    ]),

    # All-Items
    html.Div([
        # Select Model
        html.Div([
            html.Label('Dynamic Model:'),
            dcc.Dropdown(
                id = 'model',
                options=[
                    {'label': 'Black-Scholes', 'value': 'mod1'},
                ],
                value='mod1',
                style={'width':'60%'},
            ),

            html.Label('Option Type:'),
            dcc.Dropdown(
                id = 'type',
                options=[
                    {'label': 'Call', 'value': 'c'},
                    {'label': 'Put', 'value': 'p'},
                ],
                value='c',
                style={'width':'60%'},
            ),

            html.Label('S0:'),
            dcc.Input(
                placeholder='Enter asset price...',
                type='text',
                value=''
            ),

            html.Label('K:'),
            dcc.Input(
                placeholder='Enter strike price...',
                type='text',
                value=''
            ),

            html.Label('T:'),
            dcc.Input(
                placeholder='Enter maturity...',
                type='text',
                value=''
            ),

            html.Label('Sigma:'),
            dcc.Input(
                placeholder='Enter volatility...',
                type='text',
                value=''
            ),

            html.Label('Rf:'),
            dcc.Input(
                placeholder='Enter risk free rate...',
                type='text',
                value=''
            ),

            html.Label('delta:'),
            dcc.Input(
                placeholder='Enter dividend rate...',
                type='text',
                value=''
            ),
        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id = 'yc_graph',
            ),
        ], className='six columns',)
    ], className='row'),
])

if __name__ == '__main__':
    app.run_server(debug=True)
