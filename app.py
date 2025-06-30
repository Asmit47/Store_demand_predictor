from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Load the Grid Search model (ensure the model file is in the same directory)
model = pickle.load(open('model/grid_search_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get the input values from the form
            store_id = request.form['store_id']
            product_id = request.form['product_id']
            input_features = [
                float(request.form['units_sold']),
                float(request.form['units_ordered']),
                float(request.form['price']),
                float(request.form['discount']),
                float(request.form['promotion']),
                float(request.form['competitor_pricing']),
                float(request.form['month']),
                float(request.form['day']),
            ]

            # One-hot encode Store ID and Product ID
            store_ids = ['S001', 'S002', 'S003', 'S004', 'S005']
            product_ids = [f'P{str(i).zfill(3)}' for i in range(1, 21)]

            for sid in store_ids:
                input_features.append(1.0 if store_id == sid else 0.0)

            for pid in product_ids:
                input_features.append(1.0 if product_id == pid else 0.0)

            # One-hot encode other categorical features
            seasons = ['Autumn', 'Spring', 'Summer', 'Winter']
            weather_conditions = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']
            regions = ['East', 'North', 'South', 'West']

            for season in seasons:
                input_features.append(1.0 if request.form['seasonality'] == season else 0.0)

            for condition in weather_conditions:
                input_features.append(1.0 if request.form['weather_condition'] == condition else 0.0)

            for region in regions:
                input_features.append(1.0 if request.form['region'] == region else 0.0)

            # Make prediction
            prediction = model.predict([input_features])

            return render_template('predict.html', prediction_text=f'Predicted Demand: {prediction[0]}')
    except Exception as e:
        # Log the error and return a message
        print(f"Error during prediction: {e}")
        return render_template('index.html', error_text="An error occurred during prediction. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2000)