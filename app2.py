import dash
from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children="DISTRIBUTION OF NANOPARTICLES IN A POLYMER MATRIX PREDICTION"),
    html.H2(children="Problem description"),
    html.Img(src="polymer_nanoparticle.jpg", alt="Polymer nanoparticle"),
    html.Div(className='sidebar', children=[
        html.H3("Input parameters"),
        html.Label("Interaction between polymers and nanoparticles:"),
        dcc.Slider(id="ponp-slider", min=0.0, max=2.5, step=0.01, value=0.4),
        html.Label("Interaction between nanoparticles and nanoparticles:"),
        dcc.Slider(id="npnp-slider", min=0.0, max=2.5, step=0.01, value=0.4),
        html.Label("Diameter of nanoparticles:"),
        dcc.Slider(id="d-slider", min=1, max=10, step=1, value=4),
        html.Label("Number of particles:"),
        dcc.Slider(id="phi-slider", min=0.001, max=0.1, step=0.001, value=0.02),
        html.Label("Length of polymer chain:"),
        dcc.Slider(id="cLength-slider", min=5, max=100, step=1, value=20),
        html.Div([
            html.H4('Distance Range'),
            html.Label("Minimum distance in nm:"),
            dcc.Input(id='distance-min', type='text', value='0.075'),
            html.Label("Width range in nm:"),
            dcc.Input(id='distance-range', type='text', value='150'),
        ]),
        html.Button("Predict!", id="predict-button"),  # Button to trigger prediction
    ]),
    html.Div(id="prediction-output")  # Output container for prediction results
])

# Callback to update prediction when the "Predict!" button is clicked
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("ponp-slider", "value"),
    State("npnp-slider", "value"),
    State("d-slider", "value"),
    State("phi-slider", "value"),
    State("cLength-slider", "value"),
    State("distance-min", "value"),
    State("distance-range", "value")
)
def update_prediction(n_clicks, ponp, npnp, d, phi, cLength, distance_min, distance_range):
    if n_clicks is None:
        return None

    # Convert input strings to numeric values
    distance_min = float(distance_min)
    distance_range = float(distance_range)

    # Generate the input features DataFrame
    num_points = 2000
    Po_NP = [ponp] * num_points
    NP_NP = [npnp] * num_points
    D_aim = [d] * num_points
    Phi = [phi] * num_points
    Chain_length = [cLength] * num_points
    distance = np.linspace(distance_min, distance_min + distance_range, num_points)

    features = pd.DataFrame({
        'Po_NP': Po_NP,
        'NP_NP': NP_NP,
        'D_aim': D_aim,
        'Phi': Phi,
        'Chain length': Chain_length,
        'distance': distance
    })

    # Load the model and make predictions
    try:
        model = joblib.load('/mount/src/app/model.pkl')
        predictions1 = model.predict(features)

        # Create and display the prediction plot
        fig, ax = plt.subplots()
        ax.scatter(features['distance'], predictions1)
        ax.set_xlabel('distance')
        ax.set_ylabel('density')
        ax.set_title('Prediction')

        # Display the plot in Dash
        return dcc.Graph(
            id='example-graph',
            figure={'data': [{'x': features['distance'], 'y': predictions1, 'type': 'scatter'}]}
        )
    except Exception as e:
        return f"An error occurred while making predictions: {e}"

if __name__ == '__main__':
    app.run_server(debug=True)
