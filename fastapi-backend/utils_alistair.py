#Ulils_alistair.py: Utiliies desgined for Alistair Schillert: 
import pandas as pd
import os 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import timedelta
import holidays
from math import radians, sin, cos, sqrt, atan2
from fastapi import HTTPException
from datetime import datetime
import pickle 
import __main__



#######################
#
# CUSTOM PIPELINE FUNCTION
#
#
######################
##################
#Hard-Coded Data Variables
#This is used as a backup to CSV files: 
##################
# Define the timezone mapping for each IATA code


#################
#Hard-Coded Data Variables
#This is used as a backup to CSV files: 
##################
# Define the timezone mapping for each IATA code
iata_timezone_mapping = {
    'ATL': 'America/New_York',
    'LAX': 'America/Los_Angeles',
    'DFW': 'America/Chicago',
    'DEN': 'America/Denver',
    'ORD': 'America/Chicago',
    'JFK': 'America/New_York',
    'CLT': 'America/New_York',
    'MIA': 'America/New_York',
    'EWR': 'America/New_York',
    'SFO': 'America/Los_Angeles',
    'BOS': 'America/New_York',
    'LGA': 'America/New_York',
    'DTW': 'America/Detroit',
    'PHL': 'America/New_York',
    'IAD': 'America/New_York',
    'OAK': 'America/Los_Angeles'
}

# Define the DataFrame with coordinates
coordinates_df = pd.DataFrame({
    'IATA': ['ORD', 'BOS', 'LAX', 'LGA', 'ATL', 'DFW', 'JFK', 'EWR', 'MIA', 'CLT', 'SFO', 'DEN', 'DTW', 'PHL', 'IAD', 'OAK'],
    'latitude': [41.9786, 42.3656, 33.9416, 40.7769, 33.6407, 32.8998, 40.6413, 40.6895, 25.7959, 35.2140, 37.6213, 39.8561, 42.2124, 39.8744, 38.9531, 37.7126],
    'longitude': [-87.9048, -71.0096, -118.4085, -73.8740, -84.4277, -97.0403, -73.7781, -74.1745, -80.2870, -80.9431, -122.3790, -104.6737, -83.3534, -75.2424, -77.4565, -122.2197]
})

#These are unchanging coordinates, that are used to track the tine zones 

class FlightTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the US holidays for checking
        self.us_holidays = holidays.US()
        
    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self
    
    def transform(self, X):
        # Ensure flight_date and flight_time are in datetime format
        X = X.copy()
        X['flight_datetime'] = pd.to_datetime(X['flight_date'] + ' ' + X['flight_time'])
        
        # Extract year, month, day, hour, minute
        X['year'] = X['flight_datetime'].dt.year
        X['month'] = X['flight_datetime'].dt.month
        X['day'] = X['flight_datetime'].dt.day
        X['hour'] = X['flight_datetime'].dt.hour
        X['minute'] = X['flight_datetime'].dt.minute
        
        # Define season as an integer variable to condense the feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 1  # Winter
            elif month in [3, 4, 5]:
                return 2  # Spring
            elif month in [6, 7, 8]:
                return 3  # Summer
            else:
                return 4  # Autumn
        
        X['season'] = X['month'].apply(get_season)
        
        # Add US holiday flag (exact match only, no surrounding days)
        X['is_us_holiday'] = X['flight_datetime'].dt.date.apply(lambda date: date in self.us_holidays)
        # Convert boolean to integer
        X['is_us_holiday'] = X['is_us_holiday'].astype(int)
        # Drop the flight_datetime, flight_date, and flight_time columns
        X.drop(columns=['flight_datetime', 'flight_date', 'flight_time'], inplace=True)
        
        return X

class CabinTypeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Define the encoding dictionary
        self.cabinType_encoder = {
            'coach': 0,
            'business': 1,
            'premium coach': 2,
            'first': 3
        }
        
    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self
    
    def transform(self, X):
        # Make a copy of the input DataFrame to avoid modifying the original data
        X = X.copy()
        
        # Apply the encoding to the 'cabin_type' column
        X['cabin_type_ordinal'] = X['cabin_type'].map(self.cabinType_encoder)
        
        # Drop the original 'cabin_type' column
        X = X.drop(columns=['cabin_type'])
        
        return X

# Define the custom AirportDistanceGenerator class with DataFrame input for coordinates
class AirportDistanceGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, coordinates_df, airport1_col='airport1', airport2_col='airport2'):
        """
        Parameters:
        - coordinates_df: DataFrame with IATA, latitude, and longitude for airports.
        - airport1_col: Column name for the first airport IATA code in the dataframe.
        - airport2_col: Column name for the second airport IATA code in the dataframe.
        """
        self.coordinates_df = coordinates_df
        self.airport1_col = airport1_col
        self.airport2_col = airport2_col
        self.iata_coordinates = self._load_iata_coordinates()
    
    def _load_iata_coordinates(self):
        """Converts the coordinates DataFrame to a dictionary."""
        return {row['IATA']: (row['latitude'], row['longitude']) for _, row in self.coordinates_df.iterrows()}
    
    def _get_coordinates(self, iata_code):
        """Retrieve coordinates for a given IATA code from the dictionary."""
        return self.iata_coordinates.get(iata_code, (None, None))
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate the Haversine distance between two coordinates."""
        if coord1 == (None, None) or coord2 == (None, None):
            return None
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        return round(R * c,2)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Apply the distance calculation
        X['airport_distance'] = X.apply(lambda row: self._haversine_distance(
            self._get_coordinates(row[self.airport1_col]),
            self._get_coordinates(row[self.airport2_col])
        ), axis=1)
        return X

class AirportPopulationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, airport_columns=('IATA',), include_rank=True, include_passenger_count=True, include_log_transform=True):
        """
        Parameters:
        - airport_columns (tuple): Column names in the input DataFrame that correspond to airport codes.
        - include_rank (bool): Whether to include the airport's rank in the transformed output.
        - include_passenger_count (bool): Whether to include the 2023 passenger count.
        - include_log_transform (bool): Whether to include the log-transformed 2023 passenger count.
        """
        self.airport_columns = airport_columns
        self.include_rank = include_rank
        self.include_passenger_count = include_passenger_count
        self.include_log_transform = include_log_transform
        self.airport_data = self._load_airport_data()

    def _load_airport_data(self):
        """Loads airport data with rank and passenger counts for 2023."""
        return pd.DataFrame({
            'IATA': ['ATL', 'LAX', 'DFW', 'DEN', 'ORD', 'JFK', 'CLT', 'MIA', 'EWR', 'SFO', 'BOS', 'LGA', 'DTW', 'PHL', 'IAD', 'OAK'],
            'Rank': [1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 16, 19, 20, 21, 26, 43],
            '2023_Passengers': [50950068, 40956673, 39246212, 37863967, 35843104, 30804355, 25896224, 24717048, 
                                24575320, 24191159, 19962678, 16173073, 15378601, 13656189, 12073571, 5520812]
        })

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # For each airport column specified in airport_columns, merge airport data
        for airport_col in self.airport_columns:
            # Merge airport data based on the current airport column name
            merged_data = X[[airport_col]].merge(self.airport_data, how='left', left_on=airport_col, right_on='IATA')
            
            # Prepare new column names based on the airport column being processed
            new_columns = {}
            if self.include_rank:
                new_columns['Rank'] = f'{airport_col}_Rank'
            if self.include_passenger_count:
                new_columns['2023_Passengers'] = f'{airport_col}_2023_Passengers'
            if self.include_log_transform:
                merged_data['Log_2023_Passengers'] = np.log1p(merged_data['2023_Passengers'])
                new_columns['Log_2023_Passengers'] = f'{airport_col}_Log_2023_Passengers'
            
            # Rename and merge selected columns back into the main DataFrame
            X = X.join(merged_data[list(new_columns.keys())].rename(columns=new_columns))
        
        return X

#This gives the Airport surrounding cities overall "wealth"
class AirportWealthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, airport_columns=('IATA',), include_gdp=True, include_median_house=True, include_log_transform=False):
        """
        Parameters:
        - airport_columns (tuple): Column names in the input DataFrame that correspond to airport codes.
        - include_gdp (bool): Whether to include the GDP per capita in the transformed output.
        - include_median_house (bool): Whether to include the median house price.
        - include_log_transform (bool): Whether to include the log-transformed GDP and median house price.
        """
        self.airport_columns = airport_columns
        self.include_gdp = include_gdp
        self.include_median_house = include_median_house
        self.include_log_transform = include_log_transform
        self.airport_data = self._load_airport_data()

    def _load_airport_data(self):
        """Loads airport data with GDP per capita and median house prices."""
        return pd.DataFrame({
            'IATA': ['ATL', 'LAX', 'DFW', 'DEN', 'ORD', 'JFK', 'CLT', 'MIA', 'EWR', 'SFO', 
                     'BOS', 'LGA', 'DTW', 'PHL', 'IAD', 'OAK'],
            'gdp_pc': [54557, 86532, 58725, 85246, 63500, 100806, 65000, 60966, 100806, 144633, 
                       108506, 100806, 54172, 66596, 95593, 144633],
            'median_house': [359892, 953501, 499900, 684700, 325000, 732594, 400000, 558873, 
                             732594, 1236502, 718233, 732594, 277000, 250000, 1057671, 780188]
        })

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # For each airport column specified in airport_columns, merge airport data
        for airport_col in self.airport_columns:
            # Merge airport data based on the current airport column name
            merged_data = X[[airport_col]].merge(self.airport_data, how='left', left_on=airport_col, right_on='IATA')
            
            # Prepare new column names based on the airport column being processed
            new_columns = {}
            if self.include_gdp:
                new_columns['gdp_pc'] = f'{airport_col}_gdp_pc'
            if self.include_median_house:
                new_columns['median_house'] = f'{airport_col}_median_house_price'
            if self.include_log_transform:
                if self.include_gdp:
                    merged_data['Log_gdp_pc'] = np.log1p(merged_data['gdp_pc'])
                    new_columns['Log_gdp_pc'] = f'{airport_col}_Log_gdp_pc'
                if self.include_median_house:
                    merged_data['Log_median_house'] = np.log1p(merged_data['median_house'])
                    new_columns['Log_median_house'] = f'{airport_col}_Log_median_house'
            
            # Rename and merge selected columns back into the main DataFrame
            X = X.join(merged_data[list(new_columns.keys())].rename(columns=new_columns))
        
        return X

class DropAirportTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        """
        A transformer that drops specified columns from a DataFrame.
        
        Parameters:
        - columns_to_drop (list of str): The list of column names to drop. 
          If None, the columns can be specified when calling transform.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # No fitting necessary, so we return self
        return self

    def transform(self, X, columns_to_drop=None):
        """
        Drops the specified columns from the input DataFrame.
        
        Parameters:
        - X (pd.DataFrame): The input DataFrame.
        - columns_to_drop (list of str, optional): The list of columns to drop. 
          Overrides the columns_to_drop set during initialization if provided.
        
        Returns:
        - pd.DataFrame: The DataFrame with specified columns removed.
        """
        # Make a copy of the DataFrame to avoid changing the original data
        X = X.copy()
        
        # Use columns_to_drop from the method parameter if provided, otherwise fall back to instance attribute
        columns = columns_to_drop if columns_to_drop is not None else self.columns_to_drop
        
        if columns:
            # Drop the specified columns, ignoring any missing columns
            X = X.drop(columns=columns, errors='ignore')
        
        return X
 

# Attach all classes to `__main__`
def register_classes_in_main():
    __main__.FlightTimeTransformer = FlightTimeTransformer
    __main__.CabinTypeEncoder = CabinTypeEncoder
    __main__.AirportDistanceGenerator = AirportDistanceGenerator
    __main__.AirportPopulationTransformer = AirportPopulationTransformer
    __main__.AirportWealthTransformer = AirportWealthTransformer
    __main__.DropAirportTransformer = DropAirportTransformer

# Call this function to register the classes before loading the pickle
register_classes_in_main()



def run_pipeline(origin_airport, destination_airport, cabin_type, flight_date, flight_time):
    if isinstance(flight_time, str):
        time_parts = flight_time.split(':')
        if len(time_parts) == 2:
            flight_time = f"{time_parts[0]}:{time_parts[1]}:00"
        elif len(time_parts) != 3:
            raise ValueError("flight_time must be in 'HH:MM' or 'HH:MM:SS' format")
    else:
        raise ValueError("flight_time must be a string")

    # Create the DataFrame with flight details
    df = pd.DataFrame({
        'origin_airport': [origin_airport],
        'destination_airport': [destination_airport],
        'cabin_type': [cabin_type],
        'flight_date': [flight_date],        
        'flight_time': [flight_time]
    })

    try:
        with open('./models/alistair/pipeline_airport_alistair_deployment.pkl', 'rb') as file:
            alistair_pipeline = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Pipeline file not found at specified path.")
    
    # Transform the DataFrame using the pipeline
    df_transformed = alistair_pipeline.transform(df)
    return df_transformed