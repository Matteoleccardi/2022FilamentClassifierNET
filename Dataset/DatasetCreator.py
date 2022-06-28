'''
    Main dataset generator.
    To create the dataset, select how many files to generate.

'''
import numpy as np
from DatasetCreator_utils import build_main_dataset, visualize_dataset_statistics
from DatasetCreator_utils import makefilament_classes_demo, visualize_dataset_3D


# Build dataset
N_FILES = {
    "synthetic": 1,
    "cat08": 0
}
OPTIONS = "new" # "new", "add"
np.random.seed(0)
build_main_dataset(N_FILES, OPTIONS)


# Example filament
if 1:
    p = makefilament_classes_demo()
    visualize_dataset_3D(p)