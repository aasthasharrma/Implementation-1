from flask import Flask, request, jsonify
import torch
import requests
from test import REITModel  # Import the trained model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = REITModel(input_dim=4)
model.load_state_dict(torch.load("reit_model.pth"))
model.eval()

# Zillow API URL and key
ZILLOW_API_URL = "https://www.zillow.com/webservice/GetDeepSearchResults.htm"
ZILLOW_API_KEY = "YOUR_ZILLOW_API_KEY"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = torch.tensor([[
        data['inside_directors'],
        data['reit_size'],
        data['lifecycle_stage'],
        data['sp500_status']
    ]], dtype=torch.float32)
    prediction = model(features).item()
    return jsonify({"predicted_roa_roe": prediction})

@app.route('/zillow', methods=['GET'])
def get_property_data():
    address = request.args.get('address')
    citystatezip = request.args.get('citystatezip')

    # Zillow API request
    params = {
        "zws-id": ZILLOW_API_KEY,
        "address": address,
        "citystatezip": citystatezip
    }
    response = requests.get(ZILLOW_API_URL, params=params)
    return response.text  # Forward Zillow API response to the frontend

if __name__ == '__main__':
    app.run(debug=True)
