'''
File: routes.py

Purpose:
    This script contains the backend build using fastapi. All the needed api endpoints would be listed here.


'''

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import sklearn
from datetime import datetime
import utils
import utils_alistair
import re
from src.data.lookup import distance_duration_look_up
from src.features.custom_transformers import FlightTimeTransformer, CabinTypeEncoder, AirportDistanceGenerator, AirportPopulationTransformer, AirportWealthTransformer, DropAirportTransformer

app = FastAPI()


def validate_date_format(date_str: str) -> bool:
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))

#Prediction Model for Alistair Schillert's Direct Route Model: 
with open('./models/alistair/lightgbm_tuned2_alistair_alistairschillert.pkl', 'rb') as f:
    prediction_model_alistair = pickle.load(f)

with open('./models/dipesh/shrestha_dipesh_light_GBM.pkl', 'rb') as f:
    prediction_model = pickle.load(f)

@app.get("/")
def read_root():
    return {"data": "Hello world"}, 200


# function to predict fare
def predict_fare_dipesh(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    
    if not validate_date_format(flight_date):
        raise HTTPException(status_code=400, detail="Date must be in 'yyyy-mm-dd' format")
    
    predicted_fare_price = []
    
    for entry in distance_duration_look_up(origin_airport, destination_airport):

        # Prepare the input data for prediction
        input_data = utils.extract_features(origin_airport, destination_airport, 
                                            cabin_type,flight_date,flight_time, entry['nStops'], 
                                            entry["totalTravelDistance"], entry["travelDurationMinutes"])
        # Predict the fare
        try:
            
            predicted_fare_price.append({"nStops": entry['nStops'], "predicted_fare":round(prediction_model.predict(input_data)[0],2)})
            # predicted_fare_price = round(prediction_model.predict(input_data)[0],2)
            # predicted_fare_price = prediction_model.predict(input_data)[0]
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting sales: {str(e)}")

    # Return the prediction
    return {"prediction": predicted_fare_price}
    
def predict_fare_alistair(origin_airport: str, destination_airport: str, cabin_type: str, flight_date: str, flight_time: str):
    if not validate_date_format(flight_date):
        raise HTTPException(status_code=400, detail="Date must be in 'yyyy-mm-dd' format")
    predicted_fare_price = []
    try:
        # Transform data
        transformed_data = utils_alistair.run_pipeline(
            origin_airport, destination_airport, cabin_type, flight_date, flight_time
        )
        
        # Validate the transformed data shape
        if transformed_data.shape[1] != 17:
            raise ValueError("Input data shape does not match model requirements.")
        
        # Run prediction
        predicted_value = prediction_model_alistair.predict(transformed_data)
        predicted_fare_price.append({
            "nStops": 0,
            "predicted_fare": round(predicted_value[0], 2)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting fare: {str(e)}")
    
    return {"prediction": predicted_fare_price}

@app.get("/predict")
def get_fare(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    #Gets the prediction for Alistair's Model: 
    return predict_fare_alistair(origin_airport, destination_airport, cabin_type,flight_date,flight_time)
