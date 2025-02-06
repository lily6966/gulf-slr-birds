# eBird experiments dataset

This directory contains the code for preprocessing train and test datasets.

- **preprocess_folder.py** : This script creates the directory structure required for training and testing.

- **gen_data_split.sh** : This script is used to preprocess both the eBird data and map inference data. The eBird data is split into training, validation, and test sets, and its corresponding index is also be stored.

- **stemflow** : This package was used for generating spatial boxes for each month using script generating_box.py