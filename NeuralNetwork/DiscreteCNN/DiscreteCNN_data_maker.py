'''

'''
import numpy as np
import os
from DiscreteCNN import preprocess

# Manage directories
if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
    print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
    quit()

# Make the training and validation data
base_dir = os.path.join(os.getcwd(), "Dataset")
dataset_dir_list = ["synthetic_dataset", "CAT08_dataset"]
for dataset_dir in dataset_dir_list:
    for f_csv in [ f for f in os.listdir(os.path.join(base_dir, dataset_dir)) if f.endswith(".csv") ]:
        content = np.loadtxt(os.path.join(base_dir, dataset_dir, f_csv), delimiter=",", skiprows=1)
        # Get info

        # Make volumetric data

        # Create Training CSV

        # Create Validation CSV

        # Create Testing CSV
        # ... option not ready yet ...
