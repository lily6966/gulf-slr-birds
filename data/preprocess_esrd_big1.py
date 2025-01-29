import numpy as np
import os
import pickle
from sklearn import preprocessing

# File paths
ebird_file_path = "./ebird_occurance_habitat.csv"
esrd_file_path = "./pland_elev_north_east_slope_ocean_chl_poc_t_trange_3km.csv"

if __name__ == "__main__":
    # Load necessary data
    with open("feature_lst.pkl", "rb") as f:
        feature_lst = pickle.load(f)

    with open("generated_box_tuple6.pkl", "rb") as f:
        generated_box = pickle.load(f)

    # Month-day mapping
    mon2day = {1: 15, 2: 45, 3: 74, 4: 105, 5: 135, 6: 166, 7: 196, 8: 227, 9: 258, 10: 288, 11: 319, 12: 349}

    print("Box number:", len(generated_box))
    os.makedirs("./small_esrd/", exist_ok=True)  # Ensure output directory exists

    # Iterate over months
    for cnt in range(1, 13):
        mon = cnt
        features = [[] for _ in range(399)]
        labels = [[] for _ in range(399)]
        locs = [[] for _ in range(399)]

        # Process ESRD file
        with open(esrd_file_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # Skip header line

                line = line.strip().split(",")
                lat = float(line[3])
                lon = float(line[2])
                feature = np.zeros(33)
                label = np.zeros(404)
                loc = [lat, lon]
                feature[1] = mon2day[mon]

                # Reset case for each line
                case = []

                # Match with generated_box
                for sequence, lon_min, lon_max, lat_min, lat_max, day_start, day_end in generated_box:
                    if lon_min <= lon < lon_max and lat_min <= lat < lat_max and day_start <= feature[1] < day_end:
                        case.append(sequence)

                # Skip if no matching cases
                if not case:
                    continue

                # Fill feature vector
                for j, idx in enumerate(feature_lst):
                    feature[6 + j] = float(line[idx])

                # Add additional features
                feature[0] = 0
                feature[2] = 7.0
                feature[3] = 2.0
                feature[4] = 1.0
                feature[5] = 1.0
            

                # Append data to respective cases
                for item in case:
                    item = int(item)
                    features[item - 1].append(feature)
                    labels[item - 1].append(label)
                    locs[item - 1].append(np.array(loc))

        # Process and save data for each case
        for item in range(len(features)):
            if len(features[item]) == 0:
                continue

            try:
                features[item] = np.array(features[item])
                labels[item] = np.array(labels[item])
                locs[item] = np.array(locs[item])
          
                # Normalize features
                features[item] = preprocessing.scale(features[item])

                # Print sizes for debugging
                print(f"Month {mon}, Box {item + 1}")
                print(f"Features size: {features[item].shape}")
                print(f"Labels size: {labels[item].shape}")
                print(f"Locations size: {locs[item].shape}")

                # Combine data and save
                data = np.concatenate((locs[item], labels[item], features[item]), axis=1)
                print(f"Data size: {data.shape}")
                np.save(f"./small_esrd/small_esrd_{item + 1}_{mon}.npy", data)

            except Exception as e:
                print(f"Error processing Month {mon}, Box {item + 1}: {e}")
