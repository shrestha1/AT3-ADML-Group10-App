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
import re
from src.data.lookup import distance_duration_look_up
import joblib
from utils import create_input_dataframe

app = FastAPI()


def validate_date_format(date_str: str) -> bool:
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


model = joblib.load('models/sagar/final_lgbm_model.joblib')

@app.get("/")
def read_root():
    return {"data": "Hello world"}, 200


# function to predict fare
def predict_fare(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    
    departure_day = datetime.strptime(flight_date, "%Y-%m-%d")
    departure_time = datetime.strptime(flight_time, "%H:%M")

    input_df = create_input_dataframe(origin_airport, destination_airport, departure_day, departure_time, cabin_type)
    predictions = model.predict(input_df)

    stop_labels = ['0 stops', '1 stop', '2 stops', '3 stops']
    results = {stop_labels[i]: f"${predictions[i]:,.2f}" for i in range(len(predictions))}

    # Return the results as a dictionary
    return results
    



@app.get("/predict")
def get_fare_prediction(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    
    return predict_fare(origin_airport, destination_airport, cabin_type,flight_date,flight_time)
