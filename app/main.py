import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import datetime
from config import fastapi_url
from PIL import Image

# Set page configuration for light mode
st.set_page_config(page_title="AirStopShop.ai", page_icon="🛪", layout="centered" )

# Load the logo
logo_path = "airstopshop_ai.png"
logo_image = Image.open(logo_path)
try:
    st.image(logo_image, output_format="PNG")
except Exception as e:
    st.error(f"Error loading image: {e}")

st.header("Your First Stop to whether you should or shouldn't add an extra stop.")
st.text("Enter in the details of your arrival airport, destination airport, the date and time of your departure and your cabin type to get a prediction.")

# Define airport names and coordinates
airports = {
    'ATL': ('Hartsfield-Jackson Atlanta International Airport', (33.6407, -84.4277)),
    'BOS': ('Logan International Airport', (42.3656, -71.0096)),
    'CLT': ('Charlotte Douglas International Airport', (35.214, -80.9431)),
    'DEN': ('Denver International Airport', (39.8561, -104.6737)),
    'DFW': ('Dallas/Fort Worth International Airport', (32.8998, -97.0403)),
    'DTW': ('Detroit Metropolitan Airport', (42.2125, -83.3534)),
    'EWR': ('Newark Liberty International Airport', (40.6895, -74.1745)),
    'IAD': ('Washington Dulles International Airport', (38.9531, -77.4565)),
    'JFK': ('John F. Kennedy International Airport', (40.6413, -73.7781)),
    'LAX': ('Los Angeles International Airport', (33.9416, -118.4085)),
    'LGA': ('LaGuardia Airport', (40.7769, -73.8740)),
    'MIA': ('Miami International Airport', (25.7959, -80.2870)),
    'OAK': ('Oakland International Airport', (37.7213, -122.221)),
    'ORD': ('O’Hare International Airport', (41.9742, -87.9073)),
    'PHL': ('Philadelphia International Airport', (39.8744, -75.2424)),
    'SFO': ('San Francisco International Airport', (37.6213, -122.3790)),
}

st.header("Airport Selection")
st.text("Make a selection of the airport you are looking to attend!")
# Airport selection
origin_airport_code = st.selectbox("Origin Airport", options=airports.keys(), format_func=lambda x: f"{airports[x][0]} ({x})")
destination_airport_code = st.selectbox(
    "Destination Airport",
    options=[code for code in airports.keys() if code != origin_airport_code],
    format_func=lambda x: f"{airports[x][0]} ({x})"
)

# Map section
st.header("Map of Selected Route")
st.text("See your current route. Blue is your departure airport, and red is your arrival airport.")
origin_coords = airports[origin_airport_code][1]
destination_coords = airports[destination_airport_code][1]

# Initialize Folium map centered between origin and destination
map_center = ((origin_coords[0] + destination_coords[0]) / 2, (origin_coords[1] + destination_coords[1]) / 2)
folium_map = folium.Map(location=map_center, zoom_start=5)

# Add markers and labels for origin and destination airports
folium.Marker(
    location=origin_coords,
    popup=f"{airports[origin_airport_code][0]} ({origin_airport_code})",
    tooltip=f"{airports[origin_airport_code][0]} ({origin_airport_code})",
    icon=folium.Icon(color="blue", icon="plane")
).add_to(folium_map)

folium.Marker(
    location=destination_coords,
    popup=f"{airports[destination_airport_code][0]} ({destination_airport_code})",
    tooltip=f"{airports[destination_airport_code][0]} ({destination_airport_code})",
    icon=folium.Icon(color="red", icon="plane")
).add_to(folium_map)

# Add arrow line from origin to destination
folium.PolyLine(
    [origin_coords, destination_coords],
    color="black",
    weight=3,
    opacity=0.6,
    tooltip="Flight Route",
    arrow=True
).add_to(folium_map)

# Display the Folium map
st_folium(folium_map, width=700, height=500)

st.header("Departure Date and Time Selection:")
# List of cabin types
cabin_type_list = ['coach', 'first', 'premium coach', 'business']

# Flight date and time input
flight_date = st.date_input("Select Date", value=datetime.date(2022, 7, 25),
                            min_value=datetime.date(2022, 7, 15),
                            max_value=datetime.date(2023, 2, 14)
                            )
flight_time = st.time_input('Select time')

st.header("Cabin Status Selection:")
# Cabin type selection
cabin_type = st.selectbox("Cabin", cabin_type_list, index=0)

# Prediction button and API call

st.header("Predict Your Stops Pricing:")
st.text("This sends out a package to our in-house trained machine learning algorithims ready to give you a good estimate.")
if st.button('Predict'):
    headers = {'Content-Type': 'application/json'}
    url = fastapi_url + 'predict'
    
    json_data = {
        "origin_airport": origin_airport_code,
        "destination_airport": destination_airport_code,
        "cabin_type": cabin_type,
        "flight_date": flight_date.strftime('%Y-%m-%d'),
        "flight_time": flight_time.strftime("%H:%M")
    }
    
    st.write("JSON Data Sent to API:")
    st.json(json_data)

    try:
        response = requests.get(url, params=json_data, headers=headers)
        if response.status_code == 200:
            predicted_sales = response.json().get('prediction')
            st.success(f"Predicted Sales Price: {predicted_sales}")
            st.json(response.json())
        else:
            st.error(f"Error {response.status_code}: Unable to get prediction. Please check input values or try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {e}")
