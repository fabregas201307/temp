import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import html
from dash import dcc

from dash.dependencies import Input, Output, State
# import pandas_datareader.data as web # requires v0.6.0 or later
from datetime import datetime
import pandas as pd
import plotly.express as px
# from dash.dependencies import Input, Output

app = dash.Dash()

nsdq = pd.read_csv('NASDAQcompanylist.csv')
nsdq.set_index('Symbol', inplace=True)
options = []
for tic in nsdq.index:
    options.append({'label':'{} {}'.format(tic,nsdq.loc[tic]['Name']), 'value':tic})

app.layout = html.Div([
    html.H1('Airflow DAGs failures'),
    html.Div([
        html.H3('Select DAG ID:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            id='my_ticker_symbol',
            options=options,
            value=['TSLA'],
            multi=True
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'30%'}),
    html.Div([
        html.H3('Select start and end dates:'),
        dcc.DatePickerRange(
            id='my_date_picker',
            min_date_allowed=datetime(2015, 1, 1),
            max_date_allowed=datetime.today(),
            start_date=datetime(2018, 1, 1),
            end_date=datetime.today()
        )
    ], style={'display':'inline-block'}),
    html.Div([
        html.Button(
            id='submit-button',
            n_clicks=0,
            children='Submit',
            style={'fontSize':24, 'marginLeft':'30px'}
        ),
    ], style={'display':'inline-block'}),
    dcc.Graph(id='my_graph',style={'width': '90vw', 'height': '90vh'})
])

@app.callback(
    Output(component_id='my_graph', component_property='figure'),
    [Input('submit-button', 'n_clicks')],
    [State('my_ticker_symbol', 'value'),
    State('my_date_picker', 'start_date'),
    State('my_date_picker', 'end_date')])
def update_graph(n_clicks, stock_ticker, start_date, end_date):
    stock_picked = stock_ticker[0]
    piechart = px.pie(
        # nsdq[nsdq['Symbol'] == stock_picked],
        nsdq,
        values='MarketCap',
        names='Sector',
        hole=.3,
        title='Failures'
    )
    return piechart

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=8000)
