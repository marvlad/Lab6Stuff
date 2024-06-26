from dash import dcc, html, Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Read data from file
def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = [[float(x) for x in line.strip().split()] for line in lines]
    return data

# Update data for plot
def update_plot(filename, column_index, titles):
    data = read_data(filename)
    trace = go.Scatter(y=[row[column_index] for row in data], mode='lines')
    layout = go.Layout(title=titles[column_index],
                       xaxis=dict(title='Index'),
                       yaxis=dict(title='Value'))
    return {'data': [trace], 'layout': layout}

# Update data for combined plot
def update_combined_plot(filename, column_indices, titles):
    data = read_data(filename)
    traces = []
    for i, column_index in enumerate(column_indices):
        traces.append(go.Scatter(
            y=[row[column_index] for row in data],
            mode='lines',
            name=titles[column_index]
        ))
    layout = go.Layout(
        title='Combined Plot',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        legend=dict(orientation="h", x=0.1, y=1.1)
    )
    return {'data': traces, 'layout': layout}

# Main function
def main():
    filename = 'data.txt'
    titles = [
        'Humidity', 'Temperature', 'Thermistor', 'V(3.3)', 'V(3.1)',
        'V(1.8)', 'Threshold for DAC 0', 'Threshold for DAC 1', 'Saltbridge'
    ]
    # Indices of columns to plot in individual canvases
    individual_canvas_indices = [0, 1, 2, 6, 7, 8]
    # Indices of columns to plot in one canvas
    combined_canvas_indices = [3, 4, 5]

    # Create Dash app
    app = Dash(__name__)

    # Layout of the app
    app.layout = html.Div([
        html.Div([
            # Plot individual columns in separate canvases
            dcc.Graph(id=f'live-plot-{i}', figure=update_plot(filename, i, titles)) 
            for i in individual_canvas_indices
        ]),
        html.Div([
            # Plot combined columns in one canvas
            dcc.Graph(id='live-plot-combined', figure=update_combined_plot(filename, combined_canvas_indices, titles)) 
        ]),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # in milliseconds
            n_intervals=0
        )
    ])

    # Callbacks to update the individual plots
    @app.callback(
        [Output(f'live-plot-{i}', 'figure') for i in individual_canvas_indices],
        [Input('interval-component', 'n_intervals')]
    )
    def update_individual_graphs(n):
        return [update_plot(filename, i, titles) for i in individual_canvas_indices]

    # Callback to update the combined plot
    @app.callback(
        Output('live-plot-combined', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_combined_graph(n):
        return update_combined_plot(filename, combined_canvas_indices, titles)

    # Start Dash app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
