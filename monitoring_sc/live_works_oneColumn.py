import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from threading import Thread
from time import sleep

# Read data from file
def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines]
    return data

# Update data for plot
def update_plot(filename):
    data = read_data(filename)
    trace = go.Scatter(y=data, mode='lines')
    layout = go.Layout(title='Live Salt bridge {}'.format(filename),
                       xaxis=dict(title='Index'),
                       yaxis=dict(title='Resistance [Ohm] '))
    return {'data': [trace], 'layout': layout}

# Main function
def main():
    filename = 'data.txt'
    data = read_data(filename)
    plot_layout = update_plot(filename)

    # Create Dash app
    app = dash.Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        dcc.Graph(id='live-plot', figure=plot_layout),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # in milliseconds
            n_intervals=0
        )
    ])

    # Callback to update the plot
    @app.callback(
        Output('live-plot', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        return update_plot(filename)

    # Start Dash app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
