'''
    Main dataset generator.
    To create the dataset, select how many files to generate.

'''
import numpy as np
import os
from DatasetCreator_utils import build_main_dataset

if __name__ == "__main__":
    # Manage directories
    if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
        print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
        quit()
    
    # Build dataset
    N_FILES = {
        "synthetic": 15,
        "cat08": 0
    }
    OPTIONS = "new" # "new", "add"
    np.random.seed(0)
    build_main_dataset(N_FILES, OPTIONS)
