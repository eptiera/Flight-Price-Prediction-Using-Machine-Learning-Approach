from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the model and scaler
try:
    with open('best_flight_price_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('flight_price_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    logger.error(f"Error loading model or scaler files: {e}")
    raise

# Initialize label encoders and fit them with possible values
label_encoders = {}
airlines = ['Biman Bangladesh Airlines', 'US-Bangla Airlines', 'Novoair', 'Regent Airways']
sources = ['Dhaka', 'Chittagong', 'Sylhet', 'Cox\'s Bazar']
destinations = ['Dhaka', 'Chittagong', 'Sylhet', 'Cox\'s Bazar', 'Jessore', 'Rajshahi']
aircraft_types = ['Boeing 737', 'Boeing 777', 'ATR 72', 'Dash 8']
classes = ['Economy', 'Business', 'First']
booking_sources = ['Airline Website', 'Travel Agency', 'Online Travel Agency']
seasonality = ['Peak', 'Off-Peak', 'Shoulder']
stopovers = ['Non-stop', '1 Stop', '2 Stops']

# Define feature order that matches the training data
FEATURE_ORDER = [
    'Airline', 'Source', 'Destination', 'Aircraft Type', 'Class',
    'Booking Source', 'Seasonality', 'Stopovers', 'Duration (hrs)',
    'Days Before Departure'
]

# Pre-fit the label encoders with all possible values
categorical_mappings = {
    'Airline': airlines,
    'Source': sources, 
    'Destination': destinations,
    'Aircraft Type': aircraft_types,
    'Class': classes,
    'Booking Source': booking_sources,
    'Seasonality': seasonality,
    'Stopovers': stopovers
}

for feature, values in categorical_mappings.items():
    le = LabelEncoder()
    le.fit(values)
    label_encoders[feature] = le

@app.route('/')
def home():
    return render_template('index.html', 
                         airlines=airlines,
                         sources=sources,
                         destinations=destinations,
                         aircraft_types=aircraft_types,
                         classes=classes,
                         booking_sources=booking_sources,
                         seasonality=seasonality,
                         stopovers=stopovers)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        logger.debug(f"Received form data: {data}")
        
        # Validate required fields
        required_fields = ['airline', 'source', 'destination', 'aircraft_type', 
                         'class', 'booking_source', 'seasonality', 'stopovers',
                         'duration', 'days_before_departure']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                raise ValueError(f"Missing required field: {field}")
        
        # Validate numeric fields
        try:
            duration = float(data['duration'])
            days_before_departure = int(data['days_before_departure'])
            
            if duration <= 0:
                raise ValueError("Duration must be positive")
            if days_before_departure < 0:
                raise ValueError("Days before departure cannot be negative")
                
        except ValueError as e:
            logger.error(f"Numeric validation error: {str(e)}")
            raise ValueError("Invalid numeric value provided") from e
            
        # Create a DataFrame with the input data
        input_dict = {
            'Airline': data['airline'],
            'Source': data['source'],
            'Destination': data['destination'],
            'Aircraft Type': data['aircraft_type'],
            'Class': data['class'],
            'Booking Source': data['booking_source'],
            'Seasonality': data['seasonality'],
            'Stopovers': data['stopovers'],
            'Duration (hrs)': duration,
            'Days Before Departure': days_before_departure
        }
        
        # Label encode categorical features
        for col in categorical_mappings.keys():
            input_dict[col] = float(label_encoders[col].transform([input_dict[col]])[0])
        
        # Create DataFrame in correct order
        input_df = pd.DataFrame([[input_dict[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
        
        logger.debug(f"Created input DataFrame: {input_df}")
        
        # Validate categorical fields
        for col, valid_values in categorical_mappings.items():
            if input_df[col].iloc[0] not in valid_values:
                logger.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
                raise ValueError(f"Invalid value for {col}")
        
        # Ensure columns are in the correct order
        input_df = input_df[FEATURE_ORDER]
        
        logger.debug(f"Final input data shape: {input_df.shape}")
        logger.debug(f"Final input data columns: {input_df.columns}")
        
        # Scale the features
        try:
            logger.debug(f"Input data columns: {input_df.columns.tolist()}")
            logger.debug(f"Input data dtypes: {input_df.dtypes}")
            try:
                logger.debug(f"Scaler feature names: {scaler.feature_names_in_}")
            except AttributeError:
                logger.debug("Scaler does not have feature_names_in_ attribute.")
            logger.debug(f"Scaler expects shape: {scaler.mean_.shape}")
            print("Scaler expects shape:", scaler.mean_.shape)
            try:
                print("Scaler expects columns:", scaler.feature_names_in_)
            except AttributeError:
                print("Scaler does not have feature_names_in_ attribute.")
            print("Input DataFrame shape:", input_df.shape)
            print("Input DataFrame columns:", input_df.columns)
            scaled_features = scaler.transform(input_df)
            logger.debug(f"Scaled features shape: {scaled_features.shape}")
        except Exception as e:
            logger.error(f"Scaling error: {str(e)}")
            logger.error(f"Input data shape: {input_df.shape}")
            logger.error(f"Input data columns: {input_df.columns}")
            raise ValueError("Error scaling input data: Check feature order and data types") from e
        
        # Make prediction
        try:
            prediction = model.predict(scaled_features)[0]
            logger.debug(f"Prediction value: {prediction}")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError("Error making prediction") from e
        
        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })
        
    except ValueError as e:
        logger.error(f"ValueError in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        })
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        return jsonify({
            'error': f"An unexpected error occurred: {str(e)}",
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True)
