import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from boundary import call_boundary, put_boundary, call_boundary_list, put_boundary_list
from boundary_jump import call_boundary_jump, put_boundary_jump, call_boundary_jump_list, put_boundary_jump_list
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
                    {'label': 'Black-Scholes with jump', 'value': 'mod2'}
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
                    {'label': 'C', 'value': 'c'},
                    {'label': 'Gamma', 'value': 'gamma'},
                ],
                value='k',
                labelStyle={'display': 'inline-block'}
            ),

            html.Div([
                dcc.RangeSlider(
                    id='slider',
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
                    value='10'
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
                    value='0.07'
                ),
            ], className='row'),

            # Interval
            html.Div([
                dcc.Interval(
                    id='interval',
                    interval=1000,
                    n_intervals=0
                )
            ]),

            # Hidden Div1
            html.Div([
                html.Label('C:'),
                dcc.Input(
                    id = 'C',
                    placeholder='Enter jump intensity...',
                    type='text',
                    value='1'
                ),
            
                html.Label('Gamma:'),
                dcc.Input(
                    id = 'Gamma',
                    placeholder='Enter jump intensity...',
                    type='text',
                    value='0.05'
                ),
            ],id='hid')

        ], className='six columns'),
        html.Div([
            dcc.Graph(
                id = 'bound',
            ),
        ], className='six columns',)
    ], className='row'),
])

def myplot(bound):
    if len(bound.columns) == 1:
        return {
            'data': [go.Scatter(
                x = bound.index.values,
                y = bound.iloc[:,0].values,
                mode = 'lines',
            )],
            'layout': go.Layout(
                title='Exercise Boundary',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Price'},
                margin={'l': 50, 'b': 40, 't': 100, 'r': 50},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    else:
        return {
            'data': [go.Surface(
                y = bound.index.values,
                z = bound.iloc[:,1:].values,
            )],
            'layout': go.Layout(
                title='Exercise Boundary',
                autosize=True,
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

def safe_call_jump(r, delta, sigma, c, gamma, k, t):
    try:
        call_boundary_jump(r, delta, sigma, c, gamma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_put_jump(r, delta, sigma, c, gamma, k, t):
    try:
        put_boundary_jump(r, delta, sigma, c, gamma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_call_jump_list(r, delta, sigma, c, gamma, k, t):
    try:
        call_boundary_jump_list(r, delta, sigma, c, gamma, k, t)
    except Exception as e:
        print(e)
        pass

def safe_put_jump_list(r, delta, sigma, c, gamma, k, t):
    try:
        put_boundary_jump_list(r, delta, sigma, c, gamma, k, t)
    except Exception as e:
        print(e)
        pass



# run Lyasoff's code
@app.callback(
    Output('hid', 'style'),
    [Input('model', 'value'),
    Input('kind', 'value'),
    Input('K', 'value'),
    Input('T', 'value'),
    Input('Sigma', 'value'),
    Input('R', 'value'),
    Input('Delta', 'value'),
    Input('C', 'value'),
    Input('Gamma', 'value'),
    Input('slider', 'value'),
    Input('param', 'value')]
)
def run_code(model, kind, k, t, sigma, r, delta, c, gamma, slider, param):
    k = float(k)
    t = float(t)
    sigma = float(sigma)
    r = float(r)
    delta = float(delta)
    c = float(c)
    gamma = float(gamma)
    param_dict = {
        'k': k,
        't': t,
        'sigma': sigma,
        'r': r,
        'delta': delta,
        'c': c,
        'gamma': gamma,
    }
    if model == 'mod1':
        if slider[0] != slider[1]:
            if param in ['r', 'delta', 'sigma', 'gamma']:
                param_dict[param] = np.arange(float(slider[0]), float(slider[1]), 0.01)
            else:
                param_dict[param] = np.arange(float(slider[0]), float(slider[1]), 0.1)

            print(param_dict)
            print(time.time())
            if kind == 'c':
                safe_call_list(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
            elif kind == 'p':
                safe_put_list(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
            print(time.time())
        else:
            print(param_dict)
            print(time.time())
            if kind == 'c':
                safe_call(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
            elif kind == 'p':
                safe_put(param_dict['r'], param_dict['delta'], param_dict['sigma'], param_dict['k'], param_dict['t'])
            print(time.time())
    else:
        if slider[0] != slider[1]:
            if param in ['r', 'delta', 'sigma','gamma']:
                param_dict[param] = np.arange(float(slider[0]), float(slider[1]), 0.01)
            else:
                param_dict[param] = np.arange(float(slider[0]), float(slider[1]), 0.1)

            print(param_dict)
            print(time.time())
            if kind == 'c':
                safe_call_jump_list(
                    param_dict['r'], param_dict['delta'], param_dict['sigma'], 
                    param_dict['c'], param_dict['gamma'],
                    param_dict['k'], param_dict['t'])
            elif kind == 'p':
                safe_put_jump_list(
                    param_dict['r'], param_dict['delta'], param_dict['sigma'], 
                    param_dict['c'], param_dict['gamma'],
                    param_dict['k'], param_dict['t'])            
            print(time.time())
        else:
            print(param_dict)
            print(time.time())
            if kind == 'c':
                safe_call_jump(
                    param_dict['r'], param_dict['delta'], param_dict['sigma'], 
                    param_dict['c'], param_dict['gamma'],
                    param_dict['k'], param_dict['t'])
            elif kind == 'p':
                safe_put_jump(
                    param_dict['r'], param_dict['delta'], param_dict['sigma'], 
                    param_dict['c'], param_dict['gamma'],
                    param_dict['k'], param_dict['t'])
            print(time.time())
    if model == 'mod2':
        print(model)
        pass
    else:
        print(model)
        return {'display':'none'}

# update graph
@app.callback(
    Output('bound', 'figure'), 
    [Input('interval', 'n_intervals'),
    Input('slider', 'value')]
)
def update_graph(n_intervals, slider):
    while True:
        try:
            bound = pd.read_csv('bound.csv', index_col=0, header=None)
        except pd.errors.EmptyDataError as e:
            print(e)
        else:
            break
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
    Input('Delta', 'value'),
    Input('C', 'value'),
    Input('Gamma', 'value')],
)
def update_slider(param, k, t, sigma, r, delta, c, gamma):
    if param == 'k':
        k = float(k)
        slider_min = k*0.7
        slider_max = k*1.3
        return (slider_min, slider_max, [k,k])
    elif param == 't':
        t = float(t)
        slider_min = t*0.5
        slider_max = t*1.5
        return (slider_min, slider_max, [t,t])
    elif param == 'sigma':
        sigma = float(sigma)
        slider_min = 0.01
        slider_max = 0.5
        return (slider_min, slider_max, [sigma,sigma])
    elif param == 'r':
        r = float(r)
        slider_min = 0
        slider_max = 0.1
        return (slider_min, slider_max, [r,r])  
    elif param == 'delta':
        delta = float(delta)
        slider_min = 0
        slider_max = 0.1
        return (slider_min, slider_max, [delta,delta])
    elif param == 'c':
        c = float(c)
        slider_min = c*0.5
        slider_max = c*1.5
        return (slider_min, slider_max, [c,c])
    elif param == 'gamma':
        gamma = float(gamma)
        slider_min = -0.1
        slider_max = 0.1
        return (slider_min, slider_max, [gamma,gamma])

@app.callback(
    Output('slider_output', 'children'),
    [Input('slider', 'value')],
)
def update_slider_output(slider_val):
    info = f'{slider_val[0]} ~ {slider_val[1]}'
    return info

if __name__ == '__main__':
    app.run_server(debug=True)
