#########
#  file_name: lookup.py
#   Usage: Distance and duration lookup
#   Author: Dipesh
#
########

import pandas as pd

lookup_table = pd.read_csv('./models/dipesh/average_distance_duration_lookup_traindata.csv')

def distance_duration_look_up(startingAirport, destinationAirport):
    # Filter the lookup table based on the starting and destination airports
    sd_lookup = lookup_table[(lookup_table['startingAirport'] == startingAirport) & 
                             (lookup_table['destinationAirport'] == destinationAirport)]

    # Checking if the filtered result is empty
    if sd_lookup.empty:
        print("No matching records found for the given airport pair.")
        return None

    # Converting the result to a dictionary 
    return sd_lookup.to_dict('records')

if __name__ == '__main__':
    # test     
    result = distance_duration_look_up('ATL', 'LAX')
    print(result)
