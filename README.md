# Sea level rise impacts on bird distribution in the Gulf of Mexico

This project migrate code from https://github.com/gomes-lab/DMVP-DRNets/tree/v1.0 to achieve the research objective of analyzing bird population changes in the Gulf of Mexico with sea level rises.

## Initialization
To get started, install Docker and Make to work with in the development environment.
After doing so, create the container and get started training:
```bash
make build-image
...
make bash
...
root@90a65f49ddf2:/gulf-slr# 
```

## Data 
Prior to training with current dataset, ensure that you have `ebird_occurance_habitat.csv` and `pland_elev_north_east_slope_ocean_chl_poc_t_trange_3km.csv` in the `data/` folder.

## Experiment with data splitting

The dataset can be split into small overlapping regions to smooth the predictions. The splitting task was implemented using Stemflow package source code https://github.com/chenyangkang/stemflow. If this is desired, run the following:

```
# Preprocess the data 
cd data
python3 preprocess_folder.py
bash gen_data_split.sh
cd ..

# Training step
python3 train_split.py

# Evaluation test step
python3 test_split.py

# Prediction step on real-world data
python3 map_inference_split.py
```


## Log files

The training logs will be stored under **./data/train_logs/** , with filenames of the form **nosplit_ebird_x** in the nosplit case, or **ebird_y_x** in the split case, where **x** is the month and **y** is the small region number.

The nosplit evaluation test logs will be stored under **./data/test_logs/** , with filenames of the form **test_bird_x** , where **x** is the month. 

The split evaluation test logs will be stored under **./data/small_map_data[x]/**, with filenames of the form **test_bird_y_x** , where **[x]** or **x** is the month and **y** is the small region number.

The nosplit real-world test set logs will be stored under **./data/test_logs/** , with filenames of the form **test_bird_realworld_x** , where **x** is the month. 

The split real-world test set logs will be stored under **./data/small_map_data[x]/** , with filenames of the form **test_bird_realworld_y_x** , where **[x]** or **x** is the month and **y** is the small region number.

## Additional notes

- The real-world prediction step detects the best model number from the evaluation test step logs. Some regions might have a small number of data points such that no good model was found. In this case, the model number will be set to **0** or **-1**, and the total number of such regions are computed and printed as **Corrupt: xxx**. There will be corresponding error messages in the log which can be ignored, as they result from insufficient training data.

- The split test has a long runtime, so we highly recommend parallel execution.
