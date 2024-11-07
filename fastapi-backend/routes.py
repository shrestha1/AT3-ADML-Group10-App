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
import vish_utils
import re
import joblib 
from src.data.lookup import distance_duration_look_up
from src.features.custom_transformers import FlightTimeTransformer, CabinTypeEncoder, AirportDistanceGenerator, AirportPopulationTransformer, AirportWealthTransformer, DropAirportTransformer
import json 

app = FastAPI()


def validate_date_format(date_str: str) -> bool:
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))

#Prediction Model for Alistair Schillert's Direct Route Model: 
with open('./models/alistair/lightgbm_tuned2_alistair_alistairschillert.pkl', 'rb') as f:
    prediction_model_alistair = pickle.load(f)

with open('./models/dipesh/shrestha_dipesh_light_GBM.pkl', 'rb') as f:
    prediction_model_dipesh = pickle.load(f)

with open('./models/Vishwas/13796406_VS_decision_tree.joblib', 'rb') as f:
    prediction_model_vishwas = joblib.load(f)

with open('./models/sagar/final_lgbm_model.joblib', 'rb') as f:
    prediction_model_sagar = joblib.load(f)

@app.get("/")
def root():
    return {
        "Project Objectives": "This API provides the fare price forecasts for each of the given models. This connects with the AirportStopShop function.",
        "Endpoints": {
            "/": "GET: Display project objectives, list of endpoints, expected input/output, and GitHub repo link",
            "/health/": "GET: Returns status code 200 and a welcome message",
            "/predict/vish": "GET: Returns predicted fare for Vish's model",
            "/predict/alistair": "GET: Returns predicted fare for Alistair's model",
            "/predict/dipesh": "GET: Returns predicted fare for Dipesh's model",
            "/predict/sagar": "GET: Returns predicted fare for Sagar's model",
            "/predict/all_models": "GET: Returns combined fare predictions from all models",
        },
        "Expected Input/Output": {
            "/predict/vish": {
                "Input": {
                    "origin_airport": "IATA code for the origin airport",
                    "destination_airport": "IATA code for the destination airport",
                    "cabin_type": "Class of the cabin (e.g., Economy, Business)",
                    "flight_date": "Date of the flight (YYYY-MM-DD)",
                    "flight_time": "Time of the flight (HH:MM)"
                },
                "Output": {
                    "raw": {
                        "No Stops": 150.25,
                        "1 Stop": 120.50,
                        "2 Stop": 100.75,
                        "3 Stop": 90.00
                    }
                }
            },
            "/predict/alistair": {
                "Input": {
                    "origin_airport": "IATA code for the origin airport",
                    "destination_airport": "IATA code for the destination airport",
                    "cabin_type": "Class of the cabin (e.g., Economy, Business)",
                    "flight_date": "Date of the flight (YYYY-MM-DD)",
                    "flight_time": "Time of the flight (HH:MM)"
                },
                "Output": {
                    "prediction": [
                        {
                            "nStops": 0,
                            "predicted_fare": 155.75
                        }
                    ]
                }
            },
            "/predict/dipesh": {
                "Input": {
                    "origin_airport": "IATA code for the origin airport",
                    "destination_airport": "IATA code for the destination airport",
                    "cabin_type": "Class of the cabin (e.g., Economy, Business)",
                    "flight_date": "Date of the flight (YYYY-MM-DD)",
                    "flight_time": "Time of the flight (HH:MM)"
                },
                "Output": {
                    "prediction": [
                        {
                            "nStops": 0,
                            "predicted_fare": 140.45
                        },
                        {
                            "nStops": 1,
                            "predicted_fare": 115.30
                        }
                    ]
                }
            },
            "/predict/sagar": {
                "Input": {
                    "origin_airport": "IATA code for the origin airport",
                    "destination_airport": "IATA code for the destination airport",
                    "cabin_type": "Class of the cabin (e.g., Economy, Business)",
                    "flight_date": "Date of the flight (YYYY-MM-DD)",
                    "flight_time": "Time of the flight (HH:MM)"
                },
                "Output": {
                    "0 stops": 160.50,
                    "1 stop": 130.25,
                    "2 stops": 110.75,
                    "3 stops": 95.00
                }
            },
            "/predict/all_models": {
                "Input": {
                    "origin_airport": "IATA code for the origin airport",
                    "destination_airport": "IATA code for the destination airport",
                    "cabin_type": "Class of the cabin (e.g., Economy, Business)",
                    "flight_date": "Date of the flight (YYYY-MM-DD)",
                    "flight_time": "Time of the flight (HH:MM)"
                },
                "Output": {
                    "Alistair": {
                        "0 Stopover": 155.75
                    },
                    "Dipesh": {
                        "0 Stopover": 140.45,
                        "1 Stopover": 115.30
                    },
                    "Vish": {
                        "0 Stopover": 150.25,
                        "1 Stopover": 120.50,
                        "2 Stopovers": 100.75,
                        "3 Stopovers": 90.00
                    },
                    "Sagar": {
                        "0 Stopover": 160.50,
                        "1 Stopover": 130.25,
                        "2 Stopovers": 110.75,
                        "3 Stopovers": 95.00
                    }
                }
            }
        },
        "GitHub Repo": "https://github.com/shrestha1/AT3-ADML-Group10-App"
    }


@app.get('/health', status_code=200)
def healthcheck():
    return 'This API is up and running. The app AirportStopShop was created by Group 10'

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
            
            predicted_fare_price.append({"nStops": entry['nStops'], "predicted_fare":round(prediction_model_dipesh.predict(input_data)[0],2)})
            # predicted_fare_price = round(prediction_model.predict(input_data)[0],2)
            # predicted_fare_price = prediction_model.predict(input_data)[0]
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting sales: {str(e)}")

    # Return the prediction
    return {"prediction": predicted_fare_price}

# Vish function to predict fare:
def vish_pred_fare(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    df = vish_utils.vish_convert_df(origin_airport, destination_airport, cabin_type, flight_date, flight_time)
    df = vish_utils.vish_transform(df)
    df = vish_utils.vish_encode(df)
    return vish_utils.vish_predict(df)


#Alistair's Function to predict airport fare:   
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
#This function
def predict_fare_sagar(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    
    departure_day = datetime.strptime(flight_date, "%Y-%m-%d")
    departure_time = datetime.strptime(flight_time, "%H:%M")

    input_df = utils.create_input_dataframe(origin_airport, destination_airport, departure_day, departure_time, cabin_type)
    predictions = prediction_model_sagar.predict(input_df)

    stop_labels = ['0 stops', '1 stop', '2 stops', '3 stops']
    results = {stop_labels[i]: round(predictions[i], 2) for i in range(len(predictions))}

    # Return the results as a dictionary
    return results
    

@app.get("/predict/vish")
def get_fare_vish(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    #Gets the prediction for Alistair's Model: 
    return vish_pred_fare(origin_airport, destination_airport, cabin_type,flight_date,flight_time)

@app.get("/predict/alistair")
def get_fare_alistair(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    #Gets the prediction for Alistair's Model: 
    return predict_fare_alistair(origin_airport, destination_airport, cabin_type,flight_date,flight_time)

@app.get("/predict/dipesh")
def get_fare_dipesh(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    #Gets the prediction for Alistair's Model: 
    return predict_fare_dipesh(origin_airport, destination_airport, cabin_type,flight_date,flight_time)

@app.get("/predict/sagar")
def get_fare_sagar(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    #Gets the prediction for Alistair's Model: 
    return predict_fare_sagar(origin_airport, destination_airport, cabin_type,flight_date,flight_time)


####################
#
#Combining the import function for each:
#
######################

def combine_fare_predictions(alistair_data, dipesh_data, vish_data, sagar_data):
    try:
        def process_data(prediction_data, source_name):
            """Helper function to process and rename stop counts to include 'Stopovers'."""
            if not prediction_data or "prediction" not in prediction_data or not isinstance(prediction_data["prediction"], list):
                raise ValueError(f"{source_name}'s fare prediction data is missing or invalid.")
            
            # Rename keys to include 'Stopovers'
            renamed_dict = {
                f"{str(entry['nStops'])} Stopover" + ("s" if int(entry["nStops"]) > 1 else ""): entry["predicted_fare"]
                for entry in prediction_data["prediction"]
                if "nStops" in entry and "predicted_fare" in entry
            }
            
            if not renamed_dict:
                raise ValueError(f"{source_name}'s data format is invalid or missing required fields.")
            
            return renamed_dict

        # Process Alistair's data
        alistair_dict = process_data(alistair_data, "Alistair")

        # Process Dipesh's data
        dipesh_dict = process_data(dipesh_data, "Dipesh")

        # Process Vish's data
        if not vish_data or "raw" not in vish_data or not isinstance(vish_data["raw"], dict):
            raise ValueError("Vish's fare prediction data is missing or invalid.")
        
        vish_dict = {
            "0 Stopover": vish_data["raw"].get("No Stops"),
            "1 Stopover": vish_data["raw"].get("1 Stop"),
            "2 Stopovers": vish_data["raw"].get("2 Stop"),
            "3 Stopovers": vish_data["raw"].get("3 Stop")
        }
        
        if not any(vish_dict.values()):
            raise ValueError("Vish's data format is invalid or missing required stops.")

        # Process Sagar's data
        if not sagar_data or not isinstance(sagar_data, dict):
            raise ValueError("Sagar's fare prediction data is missing or invalid.")
        
        sagar_dict = {
            "0 Stopover": sagar_data.get("0 stops"),
            "1 Stopover": sagar_data.get("1 stop"),
            "2 Stopovers": sagar_data.get("2 stops"),
            "3 Stopovers": sagar_data.get("3 stops")
        }
        
        if not any(sagar_dict.values()):
            raise ValueError("Sagar's data format is invalid or missing required stops.")

        # Combine data into a dictionary
        combined_data = {
            "Alistair": alistair_dict,
            "Dipesh": dipesh_dict,
            "Vish": vish_dict,
            "Sagar": sagar_dict
        }

        # Return the combined dictionary
        return combined_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/predict/all_models")
def get_combined_fare(origin_airport: str, destination_airport: str, cabin_type: str, flight_date:str, flight_time:str):
    try:
        # Fetch predictions
        alistair_data = predict_fare_alistair(origin_airport, destination_airport, cabin_type, flight_date, flight_time)
        dipesh_data = predict_fare_dipesh(origin_airport, destination_airport, cabin_type, flight_date, flight_time)
        vish_data = vish_pred_fare(origin_airport, destination_airport, cabin_type, flight_date, flight_time)
        sagar_data = predict_fare_sagar(origin_airport, destination_airport, cabin_type, flight_date, flight_time)
        
        # Combine predictions
        combined_fare_df = combine_fare_predictions(alistair_data, dipesh_data, vish_data, sagar_data)
        return combined_fare_df
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
