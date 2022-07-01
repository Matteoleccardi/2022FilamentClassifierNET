'''

'''
import numpy as np
import os
import torch
from DiscreteCNN import preprocess

# Manage directories
if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
    print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
    quit()

# Make the training and validation data
base_dir = os.path.join(os.getcwd(), "Dataset")
dataset_dir_list = ["synthetic_dataset", "CAT08_dataset"]
traindata_save_dir = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_test_data")
classes = [0, 1, 2, 3, 4]
fnames = []
flabels = []
fcount = 0
for dataset_dir in dataset_dir_list:
    for f_csv in [ f for f in os.listdir(os.path.join(base_dir, dataset_dir)) if f.endswith(".csv") ]:
        content = np.loadtxt(os.path.join(base_dir, dataset_dir, f_csv), delimiter=",", skiprows=1)
        # Get info
        #   Class 0: all
        #   Class 1: all
        #   Class 2: max between classes 0, 1, 3, 4
        #   Class 3: all
        #   Class 4: all
        #   Class N: all il not substantially much more that the other classes
        n_limit = 0
        for c in np.delete(classes,2):
            len_c = content[content[:,3] == c,:].shape[0] 
            if len_c > n_limit:
                n_limit = len_c
        n_limit = int(2 * n_limit)
        # Make volumetric data
        for c in classes:
            n_max = np.min([(content[:,3] == c).shape[0], n_limit])
            if np.sum(content[:,3] == c) > 0:
                candidate_idxs = np.argwhere(content[:,3] == c).flatten()
                np.random.shuffle(candidate_idxs)
                for idx in candidate_idxs[:n_max]:
                    torch_tensor = preprocess(content[:,:3], idx, n_voxels_per_side=41)
                    fname = os.path.join(traindata_save_dir, f"tensor{fcount:09d}.pt")
                    torch.save(torch_tensor, fname)
                    fnames.append(fname)
                    flabels.append(c)
                    fcount += 1
                    print(f"{fcount}", end="\r")
# Setup all data
fnames = np.array(fnames)
flabels = np.array(flabels)
fdata = np.vstack((fnames, flabels)).T
n_data = fdata.shape[0]
# Create Training CSV
idx_train = int(n_data*0.7)
np.savetxt(os.path.join(traindata_save_dir, f"train.csv"), fdata[:idx_train,:], delimiter=",", header="Tensor file name,Class")
# Create Validation CSV
np.savetxt(os.path.join(traindata_save_dir, f"validation.csv"), fdata[idx_train:,:], delimiter=",", header="Tensor file name,Class")
# Create Testing CSV
# ... option not ready yet ...
print("Training and validation datasets for DiscreteCNN successfully created.")
print("Total data: ", n_data)
print("  Train data: ", idx_train)
print("  Validation data: ", n_data - idx_train)
