import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)


# Load dataset and train model
def train_model():
    # Load the dataset (replace with your actual data)
    data = pd.read_csv('air_pollution_data.csv')

    # Preprocess data
    data = data.dropna()

    # Add new features: SO2, NO2, CO, O3, PM10
    features = ['SO2', 'NO2', 'CO', 'O3', 'PM10']
    target = 'PM2.5'  # Target pollution level to predict

    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model (using RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")

    return model, scaler


# Function to classify air quality based on PM2.5
def classify_air_quality(pm25):
    if pm25 < 19:
        return "Good"
    elif 19 <= pm25 < 26:
        return "Moderate"
    elif 26 <= pm25 < 500:
        return "Unhealthy"


# Load the trained model and scaler
model, scaler = train_model()


# Route for rendering the HTML page
@app.route('/')
def home():
    return render_template('Div.html')


# Route to predict air pollution based on user input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        so2 = float(request.form['so2'])
        no2 = float(request.form['no2'])
        co = float(request.form['co'])
        o3 = float(request.form['o3'])
        pm10 = float(request.form['pm10'])

        # Prepare the input data for prediction
        input_data = np.array([[so2, no2, co, o3, pm10]])
        input_scaled = scaler.transform(input_data)

        # Predict the air pollution level (PM2.5)
        pm25_prediction = model.predict(input_scaled)

        # Classify the air quality based on PM2.5
        air_quality = classify_air_quality(pm25_prediction[0])

        # Return the predicted value and air quality
        return render_template('Div.html',
                               prediction_text=f'Predicted PM2.5: {pm25_prediction[0]:.2f} µg/m³, Air Quality: {air_quality}')

    except Exception as e:
        return render_template('Div.html', prediction_text=f"Error: {e}")


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
