import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import datetime
from config import fastapi_url
from PIL import Image
import pandas as pd
import numpy 
import plotly.graph_objects as go
import json 


# Set page configuration for light mode
st.set_page_config(page_title="AirStopShop.ai", page_icon="🛫", layout="centered" )

# CSS styling to reduce padding and margins
st.markdown("""
    <style>
    .main > div {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


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
st.text("Make a selection of the airport you are looking to attend! Choose both your origin and destination aiport, and the map will display it.")
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
st_folium(folium_map, width=800, height=600)

st.header("Departure Date and Time Selection:")
st.text("This deals is when your flight is departing from the origin airport. Enter in the date and the time.")
# List of cabin types
cabin_type_list = ['coach', 'first', 'premium coach', 'business']

# Flight date and time input
flight_date = st.date_input("Select Date", value=datetime.date(2022, 7, 25),
                            min_value=datetime.date(2022, 7, 15),
                            max_value=datetime.date(2023, 2, 14)
                            )
flight_time = st.time_input('Select time')

st.header("Cabin Status Selection:")
st.write("This is the cabin status on your ticket, such as coach, first class. Use the dropdown meanu.")
# Cabin type selection
cabin_type = st.selectbox("Cabin", cabin_type_list, index=0)

# Prediction button and API call
st.header("Your Flight Charecteristics:")
st.text("This is the data that you have entered that will be sent to our group of predicting algorithims.")
flight_characteristics = pd.DataFrame({
    "Characteristic": ["Origin Airport", "Destination Airport", "Cabin Type", "Flight Date", "Flight Time"],
    "Value": [
        origin_airport_code,
        destination_airport_code,
        cabin_type,
        flight_date.strftime('%Y-%m-%d'),
        flight_time.strftime("%H:%M")
    ]
})
# Reset the index and drop the default index column
flight_characteristics = flight_characteristics.reset_index(drop=True)
st.table(flight_characteristics)
st.header("Predict Your Stops Pricing:")
st.text("This sends out a package to our group of in-house trained machine learning algorithims ready to give you a good estimate. We compute the minimum, average and maximum from all our algorithims.")
#This button is designed to predict the prediction of all functions: 
# Prediction button and API call
if st.button('Predict'):
    headers = {'Content-Type': 'application/json'}
    url = fastapi_url + 'predict/all_models'
    
    # Define JSON data to send as query parameters in the GET request
    json_data = {
        "origin_airport": origin_airport_code,
        "destination_airport": destination_airport_code,
        "cabin_type": cabin_type,
        "flight_date": flight_date.strftime('%Y-%m-%d'),
        "flight_time": flight_time.strftime("%H:%M")
    }
    
    #st.write("JSON Data Sent to API:")
    #st.json(json_data)

    try:
        # Send JSON data as query parameters in a GET request
        response = requests.get(url, params=json_data, headers=headers)
        
        if response.status_code == 200:
            response_json = response.json()
            
            # Define columns for each stop count
            stop_columns = ["0 Stopover", "1 Stopover", "2 Stopovers", "3 Stopovers"]

            # Initialize an empty list to hold each row of data
            data_for_df = []

            # Process each key (name) in the response JSON
            for name, stops in response_json.items():
                row = {col: None for col in stop_columns}
                row['Name'] = name
                
                # Populate the row with actual fare values based on stop counts
                for stop_count, fare in stops.items():
                    standardized_stop_count = "0 Stopover" if stop_count in ["0 Stopover", "0 Stopovers"] else stop_count
                    if standardized_stop_count in stop_columns:
                        row[standardized_stop_count] = float(fare) if fare is not None else np.nan
                
                data_for_df.append(row)

            # Convert to DataFrame
            df = pd.DataFrame(data_for_df)
            df.set_index('Name', inplace=True)
            df = df[stop_columns]

            # Display the main DataFrame
            #st.success("Predicted Fare Amounts by Number of Stopovers:")
            #st.table(df)

            # Calculate statistics for each column
            df_stats = pd.DataFrame({
                "Minimum Stopover Price": df.min().round(2),
                "Average Stopover Price": df.mean().round(2),
                "Max Stopover Price": df.max().round(2)
            }, index=stop_columns).T

            # Create a display version with dollar signs
            df_stats_display = df_stats.applymap(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")

            # Display the statistics table with formatted values
            st.success("Retrived the Fare Statistics from our algorithims by Number of Stopovers:")
            st.table(df_stats_display)

            # Plot combined line graph for all statistics using Plotly
            st.subheader("Combined Stopover Price Statistics Line Graph")

            fig = go.Figure()

            # Add each statistic as a separate trace
            for stat in df_stats.index:
                fig.add_trace(go.Scatter(
                    x=stop_columns,
                    y=df_stats.loc[stat],
                    mode='lines+markers',
                    name=stat  # Label for each line in the legend
                ))

            # Update layout with titles and dollar formatting on y-axis
            fig.update_layout(
                title="Stopover Price Statistics",
                xaxis_title="Number of Stopovers",
                yaxis_title="Fare Price ($)",
                yaxis_tickprefix="$",
                height=500,
                width=800,
                legend_title="Statistics"
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"Error {response.status_code}: Unable to get prediction. Please check input values or try again.")
    
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the request: {e}")
