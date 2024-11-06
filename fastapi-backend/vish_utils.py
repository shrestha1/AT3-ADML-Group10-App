
import pandas as pd
import numpy as np
from fastapi import HTTPException
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder


from joblib import load
import math




#########
#  file_name: vish_utils.py
#   Usage: Encoding the categorical features
#   Author: Vishwas
#
########

import pandas as pd


scaler = load('models/Vishwas/13796406_scaler.joblib')
d_tree = load('models/Vishwas/13796406_VS_decision_tree.joblib')

day_mapping = {
    "Monday":1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6,
    'Sunday':7
}
cabin_mapping = {
    'coach': 1,
    'premium coach': 2,
    'business': 3,
    'first': 4
}
required_columns = [
    'isBasicEconomy', 'isNonStop', 'distance', 'dep_day_of_week', 'dep_month', 'dep_year', 'dep_hour',
    'no_of_stops', 'average_cabin_value', 'startingAirport_ATL', 'startingAirport_BOS', 'startingAirport_CLT',
    'startingAirport_DEN', 'startingAirport_DFW', 'startingAirport_DTW', 'startingAirport_EWR', 'startingAirport_IAD',
    'startingAirport_JFK', 'startingAirport_LAX', 'startingAirport_LGA', 'startingAirport_MIA', 'startingAirport_OAK',
    'startingAirport_ORD', 'startingAirport_PHL', 'startingAirport_SFO', 'destinationAirport_ATL', 'destinationAirport_BOS',
    'destinationAirport_CLT', 'destinationAirport_DEN', 'destinationAirport_DFW', 'destinationAirport_DTW',
    'destinationAirport_EWR', 'destinationAirport_IAD', 'destinationAirport_JFK', 'destinationAirport_LAX',
    'destinationAirport_LGA', 'destinationAirport_MIA', 'destinationAirport_OAK', 'destinationAirport_ORD',
    'destinationAirport_PHL', 'destinationAirport_SFO'
]


airport_coordinates = {
    'ATL': {'latitude': 33.6407, 'longitude': -84.4277},   # Hartsfield-Jackson Atlanta International Airport
    'BOS': {'latitude': 42.3656, 'longitude': -71.0096},   # Boston Logan International Airport
    'CLT': {'latitude': 35.2140, 'longitude': -80.9431},   # Charlotte Douglas International Airport
    'DEN': {'latitude': 39.8561, 'longitude': -104.6737},  # Denver International Airport
    'DFW': {'latitude': 32.8998, 'longitude': -97.0403},   # Dallas/Fort Worth International Airport
    'DTW': {'latitude': 42.2124, 'longitude': -83.3534},   # Detroit Metropolitan Wayne County Airport
    'EWR': {'latitude': 40.6895, 'longitude': -74.1745},   # Newark Liberty International Airport
    'IAD': {'latitude': 38.9531, 'longitude': -77.4565},   # Washington Dulles International Airport
    'JFK': {'latitude': 40.6413, 'longitude': -73.7781},   # John F. Kennedy International Airport
    'LAX': {'latitude': 33.9416, 'longitude': -118.4085},  # Los Angeles International Airport
    'LGA': {'latitude': 40.7769, 'longitude': -73.8740},   # LaGuardia Airport
    'MIA': {'latitude': 25.7959, 'longitude': -80.2870},   # Miami International Airport
    'OAK': {'latitude': 37.7126, 'longitude': -122.2197},  # Oakland International Airport
    'ORD': {'latitude': 41.9742, 'longitude': -87.9073},   # O'Hare International Airport
    'PHL': {'latitude': 39.8744, 'longitude': -75.2424},   # Philadelphia International Airport
    'SFO': {'latitude': 37.6213, 'longitude': -122.3790}   # San Francisco International Airport
}

def haversine_distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1['latitude']), math.radians(coord1['longitude'])
    lat2, lon2 = math.radians(coord2['latitude']), math.radians(coord2['longitude'])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def vish_dist(origin_airport, destination_airport):
    coord1 = airport_coordinates[origin_airport]
    coord2 = airport_coordinates[destination_airport]
    return haversine_distance(coord1, coord2)



def vish_convert_df(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    is_basic_economy = 1 if cabin_type == "coach" else 0
    cabin_type = cabin_mapping[cabin_type]
    distance = vish_dist(origin_airport, destination_airport)

    data = {
        'startingAirport': [origin_airport, origin_airport, origin_airport, origin_airport],
        'destinationAirport': [destination_airport, destination_airport, destination_airport, destination_airport],
        'flight_date': [flight_date, flight_date, flight_date, flight_date],
        'flight_time': [flight_time, flight_time, flight_time, flight_time],
        'isNonStop': [1, 0, 0, 0],
        'isBasicEconomy': [is_basic_economy, is_basic_economy, is_basic_economy, is_basic_economy],
        'no_of_stops': [0, 1, 2, 3],
        'average_cabin_value':[cabin_type, cabin_type, cabin_type, cabin_type],
        'distance': [distance,distance,distance,distance]
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    return df

def extract_hour(time_str):
    try:
        # Attempt to convert the time string to datetime and return the hour
        return pd.to_datetime(time_str, format='%H:%M', errors='coerce').hour
    except Exception:
        # If an error occurs, return 12
        return 12


def vish_transform(df):
    df['flight_date'] = pd.to_datetime(df['flight_date'], errors='coerce')
    df['dep_day_of_week'] = df['flight_date'].dt.day_name()  # Get the name of the day
    df['dep_month'] = df['flight_date'].dt.month  # Get the month as an integer
    df['dep_year'] = df['flight_date'].dt.year  # Get the year
    df['flight_time'] = df['flight_time'].astype(str)
    df['dep_hour'] = df['flight_time'].apply(extract_hour)
    df = df.drop(columns=['flight_date', 'flight_time' ])

    return df

def vish_encode(df):
    df['dep_day_of_week'] = df['dep_day_of_week'].map(day_mapping).astype(int)

    airport_codes = list(airport_coordinates.keys())

    # Initialize OneHotEncoder, and specify all airport codes as categories
    encoder = OneHotEncoder(categories=[airport_codes, airport_codes], sparse_output=False)

    # Apply OneHotEncoder to both 'startingAirport' and 'destinationAirport' columns
    encoded_airports = encoder.fit_transform(df[['startingAirport', 'destinationAirport']])

    # Create a DataFrame with the encoded values
    encoded_df = pd.DataFrame(encoded_airports,
                              columns=encoder.get_feature_names_out(['startingAirport', 'destinationAirport']))

    # Concatenate the encoded columns to the original DataFrame
    df_encoded = pd.concat([df, encoded_df], axis=1)
    df_encoded = df_encoded.drop(columns=['startingAirport', 'destinationAirport' ])
    df_encoded = df_encoded[required_columns]

    scaled = scaler.transform(df_encoded)

    # Convert the scaled array back to a DataFrame
    final_df = pd.DataFrame(scaled, columns=df_encoded.columns)
    return final_df

def vish_predict(df):
    #return df.to_json(orient='records')
    predictions = d_tree.predict(df)

    min_fare = round(min([predictions[0], predictions[1], predictions[2], predictions[3]]),2)

    # Create the text string, converting numbers to strings where needed
    text = "Flights available from fare starting from $" + str(min_fare) + ", with direct flights starting from $" + str(
        round(predictions[0],2))

    # Prepare the result dictionary
    pred = {
        "result": text,
        "raw": {
            "No Stops": predictions[0],
            "1 Stop": predictions[1],
            "2 Stop": predictions[2],
            "3 Stop": predictions[3],
        }
    }
    return pred