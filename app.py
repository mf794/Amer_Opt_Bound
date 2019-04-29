import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from boundary import call_boundary, put_boundary, call_boundary_list, put_boundary_list

# submit botton version

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

            html.Label('Option type:'),
            dcc.RadioItems(
                id='kind',
                options=[
                    {'label': 'Call', 'value': 'c'},
                    {'label': 'Put', 'value': 'p'},
                ],
                value='c',
                labelStyle={'display': 'inline-block'}
            ),
            
            html.Label('Change parameter:'),
            dcc.RadioItems(
                id='param',
                options=[
                    {'label': 'K', 'value': 'k'},
                    {'label': 'T', 'value': 't'},
                    {'label': 'Sigma', 'value': 'sigma'},
                    {'label': 'R', 'value': 'r'},
                    {'label': 'Dividend', 'value': 'delta'},
                ],
                value='k',
                labelStyle={'display': 'inline-block'}
            ),

            html.Div([
                dcc.RangeSlider(
                    id='slider',
                    min=0,
                    max=30,
                    value=[10, 10],
                    step=0.01,
                    allowCross=True
                )
            ]),

            html.Div(id='slider_output'),

            html.Div([
                html.Label('K:'),
                dcc.Input(
                    id = 'K',
                    placeholder='Enter strike price...',
                    type='text',
                    value='80'
                ),
            ], className='row'),


            html.Div([
                html.Label('T:'),
                dcc.Input(
                    id = 'T',
                    placeholder='Enter maturity...',
                    type='text',
                    value='1'
                ),
            ], className='row'),
        
            html.Div([
                html.Label('Sigma:'),
                dcc.Input(
                    id = 'Sigma',
                    placeholder='Enter volatility...',
                    type='text',
                    value='0.2'
                ),
            ], className='row'),
            
            html.Div([
                html.Label('R:'),
                dcc.Input(
                    id = 'R',
                    placeholder='Enter risk free rate...',
                    type='text',
                    value='0.04'
                ),
            ], className='row'),

            html.Div([
                html.Label('Dividend:'),
                dcc.Input(
                    id = 'Delta',
                    placeholder='Enter dividend rate...',
                    type='text',
                    value='0.03'
                ),
            ], className='row'),

            # Interval
            html.Div([
                dcc.Interval(
                    id='interval',
                    interval=100,
                    n_intervals=0
                )
            ]),

            # Hidden Div
            html.Div(id='hid1', style={'display':'none'})

        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id = 'bound',
            ),
        ], className='six columns',)
    ], className='row'),
])

def myplot(bound):
    if len(bound) == 2:
        return {
            'data': [go.Scatter(
                x = bound.iloc[:,0].values,
                y = bound.iloc[:,1].values,
                mode = 'lines',
            )],
            'layout': go.Layout(
                title='Exercise Boundary',
                xaxis={'title': 'Time to Maturity'},
                yaxis={'title': 'Price'},
                margin={'l': 50, 'b': 40, 't': 100, 'r': 50},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    else:
        return {
            'data': [go.Surface(
                z = bound.as_matrix(),
            )],
            'layout': go.Layout(
                title='Exercise Boundary',
                autosize=False,
                width=500,
                height=500,
                margin=dict(
                    l=65,
                    r=50,
                    b=65,
                    t=90
                )
            )
        }
def safe_call(r, delta, sigma, k, t):
    try:
        call_boundary(r, delta, sigma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_put(r, delta, sigma, k, t):
    try:
        put_boundary(r, delta, sigma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_call_list(r, delta, sigma, k, t):
    try:
        call_boundary_list(r, delta, sigma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_put_list(r, delta, sigma, k, t):
    try:
        put_boundary_list(r, delta, sigma, k, t)
    except Exception as e:
        print(e)
        pass

# run Lyasoff's code
@app.callback(
    Output('hid1', 'children'),
    [Input('model', 'value'),
    Input('kind', 'value'),
    Input('K', 'value'),
    Input('T', 'value'),
    Input('Sigma', 'value'),
    Input('R', 'value'),
    Input('Delta', 'value'),
    Input('slider', 'value'),
    Input('param', 'value')]
)
def run_code(model, kind, k, t, sigma, r, delta, slider, param):
    k = float(k)
    t = float(t)
    sigma = float(sigma)
    r = float(r)
    delta = float(delta)
    param_dict = {
        'k': k,
        't': t,
        'sigma': sigma,
        'r': r,
        'delta': delta
    }
    if slider[0] != slider[1]:
        param_dict[param] = np.arange(float(slider[0]), float(slider[1]), 0.1)
        print(param_dict)
        if kind == 'c':
            safe_call_list(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
        elif kind == 'p':
            safe_put_list(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
    else:
        print(param_dict)
        if kind == 'c':
            safe_call(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
        elif kind == 'p':
            safe_put(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])

# update graph
@app.callback(
    Output('bound', 'figure'), 
    [Input('interval', 'n_intervals'),
    Input('slider', 'value')]
)
def update_graph(n_intervals, slider):
    if slider[0] != slider[1]:
        bound = pd.read_csv('bound_list.csv', index_col=False, header=None)
    else:
        bound = pd.read_csv('bound.csv', index_col=False, header=None)
    return myplot(bound)

# update slider
@app.callback(
    [Output('slider', 'min'),
    Output('slider', 'max'),
    Output('slider', 'value')],
    [Input('param','value'),
    Input('K', 'value'),
    Input('T', 'value'),
    Input('Sigma', 'value'),
    Input('R', 'value'),
    Input('Delta', 'value'),],
)
def update_slider(param, k, t, sigma, r, delta):
    if param == 'k':
        k = float(k)
        slider_min = k*0.8
        slider_max = k*1.2
        return (slider_min, slider_max, [k,k])
    elif param == 't':
        t = float(t)
        slider_min = t*0.8
        slider_max = t*1.2
        return (slider_min, slider_max, [t,t])
    elif param == 'sigma':
        sigma = float(sigma)
        slider_min = sigma*0.8
        slider_max = sigma*1.2
        return (slider_min, slider_max, [sigma,sigma])
    elif param == 'r':
        r = float(r)
        slider_min = r*0.8
        slider_max = r*1.2
        return (slider_min, slider_max, [r,r])  
    elif param == 'delta':
        delta = float(delta)
        slider_min = delta*0.8
        slider_max = delta*1.2
        return (slider_min, slider_max, [delta,delta])

@app.callback(
    Output('slider_output', 'children'),
    [Input('slider', 'value')],
)
def update_slider_output(slider_val):
    info = f'{slider_val[0]} ~ {slider_val[1]}'
    return info

if __name__ == '__main__':
    app.run_server(debug=True)
