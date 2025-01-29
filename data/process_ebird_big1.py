import numpy as np
import pandas as pd
import os
import pickle
from sklearn import preprocessing

file_path = "./ebird_occurance_habitat.csv" #ebird with common name and habitat data

if __name__ == "__main__":

    # Load the generated box data
    with open("generated_box_tuple6.pkl", "rb") as f:
        generated_box = pickle.load(f)
    
    features = [[[] for _ in range(12)] for ii in range(400)]
    labels = [[[] for _ in range(12)] for ii in range(400)]
    locs = [[[] for _ in range(12)] for ii in range(400)]
    counts = [[[] for _ in range(12)] for ii in range(400)]
    # Read the eBird data into a DataFrame
    ebird = pd.read_csv(file_path)
    ebird = ebird.drop(columns=['year'])
    # Sum of days to determine the day ranges for each month
    sum_days = [0 for _ in range(13)]
    months = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 
    for i in range(1, 13):
        sum_days[i] = sum_days[i - 1] + months[i]

    # Initialize an empty list to store monthly DataFrames
    monthly_ebird = []

    # Iterate through each month and filter data
    for i in range(len(sum_days) - 1):
        start_day = sum_days[i]
        end_day = sum_days[i + 1]
        
        # Filter data for the current month
        monthly_df = ebird[(ebird['day_of_year'] > start_day) & (ebird['day_of_year'] <= end_day)]
        # Append the filtered DataFrame to the list
        monthly_ebird.append(monthly_df)

    # Process each monthly DataFrame
    for i, f in enumerate(monthly_ebird):
        
        mon_idx = i
        for ii, line in f.iterrows():
            if ii == 0:
                continue
        
            loc = line.iloc[1:3]
            day = float(line.iloc[4])
            loc = [float(val) for val in loc]
            lat, lon = loc[0], loc[1]
            case = []
            
            # Find the matching cases from selected_columns
            for index, (sequence, longitude_lower_bound, longitude_upper_bound, latitude_lower_bound, latitude_upper_bound, day_of_year_start, day_of_year_end) in enumerate(generated_box):
                if longitude_lower_bound <= lon < longitude_upper_bound and latitude_lower_bound <= lat < latitude_upper_bound and day_of_year_start <= day < day_of_year_end:
                    case.append(sequence)

            if len(case) == 0:
                continue
            label = line.iloc[37:]           
            feature = line.iloc[3:37].astype(float)
            for iii in range(len(feature)):
                feature.iloc[iii] = float(feature.iloc[iii])
            for iii in range(len(label)):
                label.iloc[iii] = float(label.iloc[iii])
                if label.iloc[iii] >= 1 or label.iloc[iii] < 0:
                    label.iloc[iii] = 1.
                else:
                    if label.iloc[iii] != 0:
                        raise ValueError("Strange label: %f on row %d" % (label.iloc[iii], ii))
            # Append features, labels, and other data
            for item in case:
                item = int(item) 
                features[item-1][mon_idx].append(np.array(feature))
                labels[item-1][mon_idx].append(np.array(label))
                locs[item-1][mon_idx].append(np.array(loc))
                

              
    print ("start writting to the files")

    invalid = 0

    for i in range(400):
        for j in range(12):
            features[i][j] = np.array(features[i][j])
            labels[i][j] = np.array(labels[i][j])
            locs[i][j] = np.array(locs[i][j])
            if features[i][j].shape[0] <= 0:
                invalid += 1  
                continue
            features[i][j] = preprocessing.scale(features[i][j])
            print ("Location: %d, Month %d" % (i + 1, j + 1))
            print ("Features size: ", features[i][j].shape)
            print ("labels size: ", labels[i][j].shape)
            print ("location size: ", locs[i][j].shape)
            data = np.concatenate((locs[i][j], labels[i][j], features[i][j]), axis=1)
            print (data.shape)
            np.save("./small_region/small_case%d_bird_%d.npy" % (i + 1, j + 1), data)
            #np.save("/home/shared/data4esrd/new_exp/bird_%d.npy" % (i + 1, j + 1), data)

    print ("invalid: ", invalid)