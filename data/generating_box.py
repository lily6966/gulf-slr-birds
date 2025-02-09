import os
import sys 

import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import warnings
import pickle 
import stemflow

pd.set_option('display.max_columns', None)
# warnings.filterwarnings('ignore')

from stemflow.model_selection import ST_train_test_split
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor, Generate_Quadtree
from xgboost import XGBClassifier, XGBRegressor # remember to install xgboost if you use it as base model
from stemflow.model.Hurdle import Hurdle_for_AdaSTEM, Hurdle


#load species data

df = pd.read_csv("ebird_occurance_habitat.csv")
df = df.drop(columns=['year'])#merge three years data to one year for data augmentation

#make a year data into seperate month
sum_days = [0 for _ in range(13)]

months = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 
for i in range(1, 13):
    sum_days[i] = sum_days[i-1] + months[i-1]

# Initialize an empty list to store monthly DataFrames
monthly_ebird = [0 for _ in range(12)]
print(monthly_ebird)
ebird = df
# Iterate through each month and filter data
for i in range(0, 12):
    start_day, end_day = sum_days[i], sum_days[i+1]
    
    # Filter data for the current month
    monthly_df = ebird[(ebird['day_of_year'] > start_day) & (ebird['day_of_year'] <= end_day)]
    
    # Append the filtered DataFrame to the list
    monthly_ebird[i] = monthly_df

#generate spatial box for each month 
# Initialize an empty list to store monthly DataFrames
monthly_ensembles = []
for i in range(0, 12):
    model1 = Generate_Quadtree(
    base_model=Hurdle(
        classifier=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
        regressor=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1)
    ),                                      # hurdel model for zero-inflated problem (e.g., count)
    save_gridding_plot = True,
    ensemble_fold=10,                       # data are modeled 10 times, each time with jitter and rotation in Quadtree algo
    min_ensemble_required=7,                # Only points covered by > 7 ensembles will be predicted
    grid_len_upper_threshold=10,            # force splitting if the grid length exceeds 2.5
    grid_len_lower_threshold=0.1,             # stop splitting if the grid length fall short 5         
    temporal_start=sum_days[i],                       # The next 4 params define the temporal sliding window
    temporal_end=sum_days[i+1],                            
    temporal_step=7,                       # The window takes steps of 20 DOY (see AdaSTEM demo for details)
    temporal_bin_interval=30.5,               # Each window will contain data of 30.5 DOY
    points_lower_threshold=50,              # Only stixels with more than 50 samples are trained
    Spatio1='longitude',                    # The next three params define the name of 
    Spatio2='latitude',                     # spatial coordinates shown in the dataframe
    Temporal1='day_of_year',
    use_temporal_to_train=True,             # In each stixel, whether 'DOY' should be a predictor
    n_jobs=1,                               # Not using parallel computing
    random_state=42,                        # The random state makes the gridding process reproducible
    lazy_loading=True                       # Using lazy loading for large ensemble amount (e.g., >20 ensembles). 
                                            # -- Each trained ensemble will be saved into disk and will only be loaded if needed (e.g. for prediction).
    )
    df2 = monthly_ebird[i]
    y1 = df2['Acadian Flycatcher'].values
    df2 = df2.drop(['locality_id'], axis=1)  #stemflow cannot handle character columns so remove locality_id and common_name for now
    
    model1.implement_split(df2.reset_index(drop=True), y1, verbosity=1)
    model1.ensemble_df['month'] = i+1
    model1.ensemble_df['sequence'] = model1.ensemble_df.groupby('month').cumcount()
    # Append the filtered DataFrame to the list
    monthly_ensembles.append(model1.ensemble_df)
    # Save the plot as a PDF
    month_name = pd.Timestamp(f'2024-{i+1:02d}-01').strftime('%B')  # Get month name
    model1.gridding_plot.savefig(f'{month_name}_plot5.pdf', format='pdf')



# Combine all monthly DataFrames into a single large DataFrame
ensemble_df = pd.concat(monthly_ensembles, ignore_index=True)

# Group by '4_week_group' and get the max of 'sequence' for each group
max_sequence_per_group1 = ensemble_df.groupby('month')['sequence'].max()

# Convert the result to a DataFrame (optional)
max_sequence_per_group_df1 = max_sequence_per_group1.reset_index()

# Display the box numbers for each month
print(max_sequence_per_group_df1)


#transform box jitter index into actural geo-reference coordinate
from stemflow.utils.jitterrotation.jitterrotator import JitterRotator
from shapely.geometry import Polygon
import geopandas as gpd
# Remember to install shapely and geopandas if you haven't

# define a function
def geo_grid_geometry(line):
    old_x, old_y = JitterRotator.inverse_jitter_rotate(
        [line['stixel_calibration_point_transformed_left_bound'], line['stixel_calibration_point_transformed_left_bound'], line['stixel_calibration_point_transformed_right_bound'], line['stixel_calibration_point_transformed_right_bound']],
        [line['stixel_calibration_point_transformed_lower_bound'], line['stixel_calibration_point_transformed_upper_bound'], line['stixel_calibration_point_transformed_upper_bound'], line['stixel_calibration_point_transformed_lower_bound']],
        line['rotation'],
        line['calibration_point_x_jitter'],
        line['calibration_point_y_jitter'],
    )

    polygon = Polygon(list(zip(old_x, old_y)))
    return polygon

# Make a geometry attribute for each stixel
ensemble_df['geometry'] = ensemble_df.apply(geo_grid_geometry, axis=1)
ensemble_df = gpd.GeoDataFrame(ensemble_df, geometry='geometry')

# Assuming 'ensemble_df' is your GeoDataFrame with a 'geometry' column

# Extract bounds from geometry column
ensemble_df['longitude_lower_bound'] = ensemble_df.geometry.bounds.minx
ensemble_df['longitude_upper_bound'] = ensemble_df.geometry.bounds.maxx
ensemble_df['latitude_lower_bound'] = ensemble_df.geometry.bounds.miny
ensemble_df['latitude_upper_bound'] = ensemble_df.geometry.bounds.maxy

 
# Save the tuple to a .pkl file
# Select necessary columns
selected_columns1 = ensemble_df[['month', 'sequence', 'longitude_lower_bound','longitude_upper_bound',
                                      'latitude_lower_bound', 'latitude_upper_bound', 'day_of_year_start', 'day_of_year_end']]

generated_box_tuple1 = tuple(map(tuple, selected_columns1.values))

with open('generated_box_tuple15.pkl', 'wb') as f:
    pickle.dump(generated_box_tuple1, f)