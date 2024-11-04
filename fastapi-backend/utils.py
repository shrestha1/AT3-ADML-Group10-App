# ''' data format
#['startingAirport', 'destinationAirport', 'totalTravelDistance',
#   'travelDurationMinutes', 'cabinType', 'nStops', 'flightSchedule',
#    'month', 'day']        
# '''
import pandas as pd
import numpy as np
from fastapi import HTTPException
from datetime import datetime

import pickle


#########
#  file_name: encoder.py
#   Usage: Encoding the categorical features
#   Author: Dipesh
#
########

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


airport_encoder = {
    'ATL':0, 'BOS':1, 'CLT':2, 'DEN':3, 
    'DFW':4, 'DTW':5, 'EWR':6, 'IAD':7, 
    'JFK':8, 'LAX':9, 'LGA':10, 'MIA':11, 
    'OAK':12, 'ORD':13, 'PHL':14, 'SFO':15
}

cabinType_encoder = {
    'coach':0,
    'business':1,
    'premium coach':2,
    'first': 3
}

flightSchedule_encoder = {
    'morning': 0,
    'afternoon': 1,
    'evening': 2,
    'night': 3
}

category_encoders = {
    'startingAirport': airport_encoder,
    'destinationAirport': airport_encoder,
    'cabinType': cabinType_encoder,
    'flightSchedule': flightSchedule_encoder
}


class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns=None, encoders=None):
        self.categorical_columns = categorical_columns
        self.encoders = encoders if encoders is not None else {}
    
    def fit(self, X, y=None):
        # Ensure categorical_columns is provided
        if self.categorical_columns is None:
            raise ValueError("You must provide 'categorical_columns' for the encoder to work.")
        
        return self
    
    def transform(self, X):
        # Ensure that the encoders are provided for the categorical columns
        if self.categorical_columns is None or not self.encoders:
            raise RuntimeError("You must provide encoders for the categorical columns.")
        
        # Create a copy of the original DataFrame to avoid modifying it
        X_transformed = X.copy()
        
        # Apply the custom encoding based on the provided encoders
        for col in self.categorical_columns:
            if col in self.encoders:
                # Map the values using the custom encoder
                X_transformed[col] = X[col].map(self.encoders[col]).astype(int)
            else:
                raise ValueError(f"No encoder provided for column '{col}'")
        
        return X_transformed



def catogorize_time(flight_time):
    time = datetime.strptime(flight_time, "%H:%M").time()
    if 0 <= time.hour < 12:
        return "morning"
    elif 12 <= time.hour < 18:
        return "afternoon"
    elif 18 <= time.hour < 21:
        return "evening"
    else:
        return "night"
    
def extract_date_feature(date):
    date = datetime.strptime(date, "%Y-%m-%d")
    try:
        date = pd.to_datetime(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Extract features from the date
    month = date.month
    day = date.day
    return month, day
    





# columns
categorical_columns = ['startingAirport', 'destinationAirport', 'cabinType', 'flightSchedule']
numeric_columns = ['totalTravelDistance', 'travelDurationMinutes']

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical_encoder', CustomCategoricalEncoder(categorical_columns=categorical_columns, encoders=category_encoders), categorical_columns),
        ('scaler', StandardScaler(), numeric_columns)
    ],
    remainder='passthrough'  # Leave other columns unchanged if any
)




def extract_features(origin_airport, destination_airport, cabin_type, flight_date, flight_time, 
                     nStops, totalTravelDistance, travelDurationMinutes):
    
    flightSchedule = catogorize_time(flight_time)
    month, day = extract_date_feature(flight_date)

    df = pd.DataFrame({
        'startingAirport': [origin_airport],
        'destinationAirport': [destination_airport],
        'cabinType': [cabin_type],
        'flightSchedule': [flightSchedule],        
        'totalTravelDistance': [totalTravelDistance],
        'travelDurationMinutes': [travelDurationMinutes],
        'nStops': [nStops],
        'month': [month],
        'day':[day]
    })
    
    with open('./models/dipesh/cat_encoder.pkl', 'rb') as file:
        loaded_pipeline = pickle.load(file)
    
    df_encoded = loaded_pipeline.transform(df)
    return df_encoded