## Prediction

import streamlit as st
import requests
import datetime
from config import fastapi_url


st.header("Air Fare Prediction")

# state_list = list(data.state_store.keys())

# list of airports
origin_airport_list = ['ATL', 'BOS', 'CLT', 'DEN', 'DFW', 'DTW', 'EWR', 'IAD', 'JFK', 'LAX', 'LGA', 'MIA', 'OAK', 'ORD', 'PHL', 'SFO']

# list of destination airports
destination_airport_list = set(['ATL', 'BOS', 'CLT', 'DEN', 'DFW', 'DTW', 'EWR', 'IAD', 'JFK', 'LAX', 'LGA', 'MIA', 'OAK', 'ORD', 'PHL', 'SFO'])

# list of cabins
cabin_type_list = ['coach', 'first', 'premium coach', 'business']

origin_airport = st.selectbox("Origin Airport", origin_airport_list, index=0)
destination_airport = st.selectbox("Destination Airport", destination_airport_list-set([origin_airport]), index=0)


flight_date = st.date_input("Select Date",  value = datetime.date(2022, 7, 25),
                            min_value = datetime.date(2022, 7, 15),
                            max_value = datetime.date(2023, 2, 14)
                            )

# flight_time = st.text_input("") 
flight_time = st.time_input('Select time')

# select 
cabin_type = st.selectbox("Cabin", cabin_type_list, index=0)

if st.button('Predict'):
    headers = {'Content-Type': 'application/json'}
    url = fastapi_url+'predict'
    
    json_data = {
        "origin_airport":origin_airport,
        "destination_airport":destination_airport,
        "cabin_type":cabin_type,
        "flight_date": flight_date.strftime('%Y-%m-%d'),
        "flight_time": flight_time.strftime("%H:%M")
    }
    
    st.write("Json Data Sent to API:")
    st.json(json_data)  # Neater format for JSON display

    try:
        
        response = requests.get(url, params=json_data, headers=headers)
        # Check if the response is successful
        if response.status_code == 200:
            predicted_sales = response.json().get('prediction')
            st.success(f"Predicted Sales Price: {predicted_sales}")
            st.json(response.json())
        else:
            st.error(f"Error {response.status_code}: Unable to get prediction. Please check input values or try again.")
        
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {e}")