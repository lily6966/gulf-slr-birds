#python3 train.py --model_dir ./tmpmodel --summary_dir ./tmpsum --visual_dir ./tmpvis --data_dir ../data/new_bird_1.npy --train_idx ../data/new_bird_train_idx_1.npy --valid_idx ../data/new_bird_val_idx_1.npy --test_idx ../data/new_bird_test_idx_1.npy --r_dim 500 --r_max_dim 500

import os

import numpy as np
import os

for i in range(1, 400):
    for j in range(1, 13):
        # Load the data to determine r_dim (number of bird species/response variables)
        data_path = f"./data/small_region/small_case{i}_bird_{j}.npy"
        
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)
            
            # Assuming labels start at column index 2 and features after labels
            loc_dim = 2  # Longitude and Latitude
            feature_dim = 33  # From feature extraction logic
            r_dim = data.shape[1] - (loc_dim + feature_dim)
            
            command = (
                f"python3 train.py --model_dir ./data/small_model_mon{j}/model_{i}_{j} "
                f"--summary_dir ./summary_{i}_{j} --visual_dir ./vis_{i}_{j} "
                f"--data_dir {data_path} "
                f"--train_idx ./data/small_region/small_case{i}_bird_train_idx_{j}.npy "
                f"--valid_idx ./data/small_region/small_case{i}_bird_val_idx_{j}.npy "
                f"--test_idx ./data/small_region/small_case{i}_bird_test_idx_{j}.npy "
                f"--r_dim {r_dim} --r_max_dim {r_dim} "
                f"| tee ./data/small_map_data_{j}/ebird_{i}_{j}"
            )

            print(command)
            os.system(command)
        else:
            print(f"File {data_path} not found, skipping...")


