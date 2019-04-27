import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

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

            html.Label('Option kind:'),
            dcc.RadioItems(
                id='kind',
                options=[
                    {'label': 'Call', 'value': 'c'},
                    {'label': 'Put', 'value': 'p'},
                ],
                value='c',
                labelStyle={'display': 'inline-block'}
            ),

            html.Label('K:'),
            dcc.Input(
                id = 'K',
                placeholder='Enter strike price...',
                type='text',
                value=''
            ),

            html.Label('T:'),
            dcc.Input(
                id = 'T',
                placeholder='Enter maturity...',
                type='text',
                value=''
            ),

            html.Label('Sigma:'),
            dcc.Input(
                id = 'Sigma',
                placeholder='Enter volatility...',
                type='text',
                value=''
            ),

            html.Label('R:'),
            dcc.Input(
                id = 'R',
                placeholder='Enter risk free rate...',
                type='text',
                value=''
            ),

            html.Label('Delta:'),
            dcc.Input(
                id = 'Delta',
                placeholder='Enter dividend rate...',
                type='text',
                value=''
            ),

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


# run Lyasoff's code
@app.callback(
    Output('hid1', 'children'),
    [Input('model', 'value'),
    Input('kind', 'value'),
    Input('K', 'value'),
    Input('T', 'value'),
    Input('Sigma', 'value'),
    Input('R', 'value'),
    Input('Delta', 'value'),]
)
def run_code(model, kind, k, t, sigma, r, delta):
    print(
        '''
            model={}
            kind={}
            K={}
            T={}
            Sigma={}
            R={}
            Delta={}
        '''.format(model, kind, k, t, sigma, r, delta)
    )

# update graph
@app.callback(
    Output('bound', 'figure'), 
    [Input('interval', 'n_intervals')]
)
def update_graph(n_intervals):
    bound = pd.read_csv('bound.csv', index_col=False, header=None)
    return myplot(bound)

if __name__ == '__main__':
    app.run_server(debug=True)
