from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Read data from file
def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [[float(x) for x in line.strip().split()] for line in lines]
    # Extract only the 3rd and 9th columns
    modified_data = [[row[2], row[8]] for row in data]
    return modified_data

# Update data for plot
def update_plot(filename, column_index, titles):
    data = read_data(filename)
    trace = go.Scatter(x=list(range(len(data))), y=[row[column_index] for row in data], mode='lines')
    layout = go.Layout(title=titles[column_index],
                       xaxis=dict(title='Index'),
                       yaxis=dict(title='Value'))
    return {'data': [trace], 'layout': layout}

# Main function
def main():
    filename = 'data.txt'
    titles = ['Thermistor', 'Saltbridge']  # Titles for 3rd and 9th columns

    # Create Dash app
    app = Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        html.Div([
            # Plot 3rd column
            dcc.Graph(id='live-plot-3rd', figure=update_plot(filename, 0, titles)) 
        ]),
        html.Div([
            # Plot 9th column
            dcc.Graph(id='live-plot-9th', figure=update_plot(filename, 1, titles)) 
        ]),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # in milliseconds
            n_intervals=0
        )
    ])

    # Callbacks to update the plots
    @app.callback(
        Output('live-plot-3rd', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_3rd_graph(n):
        return update_plot(filename, 0, titles)

    @app.callback(
        Output('live-plot-9th', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_9th_graph(n):
        return update_plot(filename, 1, titles)

    # Start Dash app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
