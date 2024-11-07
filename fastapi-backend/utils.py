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

import math
import joblib
import os


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



    ##################################################################################################################################################################################################
    #Sagar Thapa (24995235)

# this is sourced from http://www.gcmap.com/

airport_coords = {
    "ATL": (33.6407, -84.4279),
    "BOS": (42.3656, -71.0096),
    "CLT": (35.214, -80.9431),
    "DEN": (39.8617, -104.6738),
    "DFW": (32.8968, -97.038),
    "DTW": (42.2124, -83.3534),
    "EWR": (40.6895, -74.1745),
    "IAD": (38.9531, -77.4477),
    "JFK": (40.6413, -73.7781),
    "LAX": (33.9425, -118.4081),
    "LGA": (40.7769, -73.8719),
    "MIA": (25.7959, -80.2870),
    "OAK": (37.7213, -122.2216),
    "ORD": (41.9742, -87.9073),
    "PHL": (39.8719, -75.2411),
    "SFO": (37.6213, -122.3790),
}

def haversine(coord1, coord2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance

def airport_distance(airport1, airport2):
    if airport1 not in airport_coords or airport2 not in airport_coords:
        raise ValueError("One or both airport codes are invalid.")
    
    coord1 = airport_coords[airport1]
    coord2 = airport_coords[airport2]
    
    return haversine(coord1, coord2)
    


def create_input_dataframe(departure_airport, destination_airport, departure_day, departure_time, cabin_code):
    """
    Creates a DataFrame for model input based on departure airport, destination airport, 
    departure date, departure time, and cabin code. Outputs 4 rows for each value of number_of_transits (0, 1, 2, 3).
    The code checks the final_lookup_df for the matching combination of startingAirport, destinationAirport, 
    and number_of_transits, and adds the corresponding averageTravelDistance and averageTravelDuration.
    """
    
    # Load lookup files
    most_frequent_airline_df = pd.read_csv('models/sagar/most_frequent_airline_codes_by_time2.csv')
    final_lookup_df = pd.read_csv('models/sagar/final_lookup.csv')

    # Ensure departure_day is a datetime object (if not already)
    departure_day = pd.to_datetime(departure_day) if not isinstance(departure_day, pd.Timestamp) else departure_day

    # Validate departure_time
    departure_time = int(departure_time.strftime('%H%M'))
    
    if not (0 <= departure_time < 2400 and departure_time % 100 < 60):
        raise ValueError(f"Invalid time format: {departure_time}. Use 'HHMM' format.")

    # Convert departure_time to hours and minutes
    departure_hour = departure_time // 100
    departure_minute = departure_time % 100

    # Extract year, month, and day from departure_day
    flight_month = departure_day.month
    flight_day_of_week = departure_day.dayofweek

    # Calculate total travel distance (replace this with your actual function)
    total_travel_distance = airport_distance(departure_airport, destination_airport)

    # Define lists for airline and cabin codes
    airline_codes = ['9K', '9X', 'AA', 'AS', 'B6', 'DL', 'F9', 'HA', 'KG', 'NK', 'SY', 'UA']
    cabin_codes = ['coach', 'first', 'premium coach']

    # Prepare the list for storing results
    input_data_list = []

    # For each value of number_of_transits (0, 1, 2, 3)
    for number_of_transits in range(4):
        # Create a new row with common values, changing only number_of_transits
        input_data = pd.DataFrame({
            'travelDuration': [0],  
            'isBasicEconomy': [True],  
            'totalTravelDistance': [total_travel_distance],  # Total travel distance
            'number_of_transits': [number_of_transits],  # This will vary (0, 1, 2, 3)
            'no_of_days_to_flight': [0.0],  #ended up not being used
            'flight_month': [flight_month],  # Extracted month
            'flight_day_of_week': [flight_day_of_week],  # Day of the week as int
            'flight_hour': [departure_hour],  # Extracted hour from time
        })

        # Add airline and cabin code placeholders (set all to False initially)
        for code in airline_codes:
            input_data[f'segmentsAirlineCode_{code}'] = [False]
        for code in cabin_codes:
            input_data[f'segmentsCabinCode_{code}'] = [False]

        # **Airline Code Lookup**

        # Find matching airline code based on airport and time
        matching_airlines = most_frequent_airline_df.loc[(
            most_frequent_airline_df['startingAirport'] == departure_airport) & 
            (most_frequent_airline_df['destinationAirport'] == destination_airport) & 
            (most_frequent_airline_df['flight_hour'] == departure_hour)
        ]
        if not matching_airlines.empty:
            for _, row in matching_airlines.iterrows():
                airline_code = row['segmentsAirlineCode']
                input_data[f'segmentsAirlineCode_{airline_code}'] = [True]

        # **Cabin Code Lookup**

        if cabin_code in cabin_codes:
            input_data[f'segmentsCabinCode_{cabin_code}'] = [True]

        # **Travel Distance, Duration, and Transits Lookup**
        
        # Find matching travel details in final_lookup_df using startingAirport, destinationAirport, and number_of_transits
        matching_lookup = final_lookup_df.loc[(
            final_lookup_df['startingAirport'] == departure_airport) & 
            (final_lookup_df['destinationAirport'] == destination_airport) & 
            (final_lookup_df['number_of_transits'] == number_of_transits)
        ]
        
        if not matching_lookup.empty:
            # If there's a match, add average travel distance and duration
            input_data['averageTravelDistance'] = matching_lookup.iloc[0]['averageTravelDistance']
            input_data['averageTravelDuration'] = matching_lookup.iloc[0]['averageTravelDuration']
        else:
            # If no match found, fallback to zeros
            input_data['averageTravelDistance'] = 0
            input_data['averageTravelDuration'] = 0

        # Now, update travelDuration and totalTravelDistance with the corresponding average values
        input_data['travelDuration'] = input_data['averageTravelDuration']
        input_data['totalTravelDistance'] = input_data['averageTravelDistance']

        # Append the current row to the list
        input_data_list.append(input_data)

    # Concatenate the list of DataFrames into a single DataFrame
    input_data_final = pd.concat(input_data_list, ignore_index=True)

    # Ensure the columns are in the correct order
    input_data_final = input_data_final[['travelDuration', 'isBasicEconomy', 'totalTravelDistance', 
                                         'number_of_transits', 'flight_month', 
                                         'flight_day_of_week', 'flight_hour' 
                                         ] + 
                                         [f'segmentsAirlineCode_{code}' for code in airline_codes] + 
                                         [f'segmentsCabinCode_{code}' for code in cabin_codes]]

     # Fill missing values with the most frequent (mode) value for each column
    for column in input_data_final.columns:
    # Replace 0s with NaN (since we want to handle zeros and NaNs together)
        input_data_final[column] = input_data_final[column].replace(0, pd.NA)
    
    # Get the most frequent value (mode) for the column
        most_frequent_value = input_data_final[column].mode().iloc[0] if not input_data_final[column].mode().empty else 0
    
    # Fill NaNs with the most frequent value
        input_data_final[column].fillna(most_frequent_value, inplace=True)

    return input_data_final


    ##################################################################################################################################################################################