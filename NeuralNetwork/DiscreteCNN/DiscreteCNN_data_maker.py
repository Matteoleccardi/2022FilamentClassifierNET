'''
    Training and validation datasets creator for the network DiscreteCNN
'''
import numpy as np
import os
import torch
import time
from multiprocessing import Pool, cpu_count
from threading import Thread
from DiscreteCNN import preprocess

# Function to save volumes in multiprocessing
def make_volume(data_row_as_list):
    '''
        Input: a single np.array() row of "data"
    '''
    points = np.loadtxt(data_row_as_list[0], delimiter=",", skiprows=1, usecols=(0,1,2))
    torch_tensor = preprocess(points, int(data_row_as_list[1]), n_voxels_per_side=int(data_row_as_list[4]))
    torch.save(torch_tensor, data_row_as_list[3])

def logger(n_target:int, dir_pt):
    n = 0
    pulse = False
    while n < n_target:
        n = len([name for name in os.listdir(dir_pt) if (os.path.isfile(os.path.join(dir_pt, name)) and name.endswith(".pt"))])
        pulser = " | " if pulse else " - "
        pulse = not pulse
        print(f"{100*n/n_target:3.2f}% completed. " + pulser, end="\r")
        time.sleep(0.5)

if __name__ == '__main__':
    # Manage directories
    if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
        print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
        quit()
    base_dir = os.path.join(os.getcwd(), "Dataset")
    dataset_dir_list = ["synthetic_dataset", "CAT08_dataset"]
    traindata_save_dir = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data")

    # Dataset infos
    classes = [0, 1, 2, 3, 4]
    n_voxels_per_side=41

    # Gather all infos to then create the dataset
    fnames_fil  = [] # File name of the origin filament
    indexes_fil = [] # Index of the points to be classified
    flabels     = [] # Label of the classification (class identifier)
    fnames_vol  = [] # Name of the volume the DiscreteCNN takes as input (torch.tensor of shape (1,1,n_voxels_per_side,n_voxels_per_side,n_voxels_per_side))
    fcount = 0
    print("Gathering infos...")
    for dataset_dir in dataset_dir_list:
        for f_csv in [ f for f in os.listdir(os.path.join(base_dir, dataset_dir)) if f.endswith(".csv") ]:
            content_classes = np.loadtxt(os.path.join(base_dir, dataset_dir, f_csv), delimiter=",", skiprows=1, usecols=3).flatten()
            # Get max number of samples per class
            #   Class 0: take all
            #   Class 1: take all
            #   Class 2: take max between classes 0, 1, 3, 4
            #   Class 3: take all
            #   Class 4: take all
            #   Class N: take all if not substantially much more that the other classes
            n_limit = 0
            for c in np.delete(classes,2):
                len_c = content_classes[content_classes == c].shape[0] 
                if len_c > n_limit:
                    n_limit = len_c
            n_limit = int(1.5 * n_limit)
            # Gather infos in common-order lists
            for c in classes:
                n_max = np.min([content_classes[content_classes == c].shape[0], n_limit])
                if n_max > 0:
                    candidate_idxs = np.argwhere(content_classes == c).flatten()
                    np.random.shuffle(candidate_idxs)
                    for idx in candidate_idxs[:n_max]:
                        fnames_fil.append( os.path.abspath(os.path.join(base_dir, dataset_dir, f_csv)) )
                        indexes_fil.append(idx)
                        flabels.append(c)
                        fnames_vol.append( os.path.abspath(os.path.join(traindata_save_dir, f"tensor{fcount:09d}.pt")) )
                        fcount += 1
                        print(f"{fcount}", end="\r")
    print(f"Infos gathered. Total datapoints: {len(flabels)}")

    # Format output in convenient way to be passed to a multiple process
    n_voxels_per_side_list = (n_voxels_per_side*np.ones(len(flabels), int)).tolist()
    data = [fnames_fil, indexes_fil, flabels, fnames_vol, n_voxels_per_side_list]
    data = np.array(data).T

    # Make the volumetric data with multiprocess
    print(f"Creating torch input tensors on {cpu_count()+1} processes (= number of CPUs cores + 1)...")
    multiprocess_inputs = []
    for row in data:
        multiprocess_inputs.append(row)
    thr_logger = Thread(target=logger, args=(len(multiprocess_inputs), os.path.abspath(traindata_save_dir)) )
    thr_logger.start()
    t0 = time.time()
    with Pool(processes=cpu_count()+1) as p:
            p.map(make_volume, multiprocess_inputs)
    t1 = time.time()
    thr_logger.join()
    # Check if all data were created successfully
    missing_files = []
    for row in multiprocess_inputs:
        filename = row[3]
        if not os.path.isfile(filename):
            missing_files.append(filename)
    if len(missing_files) == 0:
        print(f"All {len(multiprocess_inputs)} files successfully created in {int(np.floor((t1-t0)/60))} min {int(t1-t0) % 60} s.")
    else:
        print(f"{len(missing_files)} files were not created (on {len(multiprocess_inputs)} files):")
        print(missing_files)

    # Put everything in CSV files
    data_to_save = []
    for n, c in zip(fnames_vol, flabels):
        data_to_save.append((n, c))
    data_to_save = np.array(data_to_save, dtype=[("fname","U256"),("class","i4")])
    n_data = data_to_save.shape[0]
    np.random.shuffle(data_to_save)
    # -> Create Training CSV
    idx_train = int(n_data*0.7)
    np.savetxt(os.path.join(traindata_save_dir, f"train.csv"), data_to_save[:idx_train], delimiter=",", header="Tensor file name,Class", fmt=["%s", "%d"])
    # -> Create Validation CSV
    np.savetxt(os.path.join(traindata_save_dir, f"validation.csv"), data_to_save[idx_train:], delimiter=",", header="Tensor file name,Class", fmt=["%s", "%d"])
    print("----------------------------------------------------------------------")
    print("Training and validation datasets for DiscreteCNN successfully created.")
    print("Folder: " + traindata_save_dir)
    print("Total data: ", n_data)
    print(" |-- Train data: ", idx_train)
    print(" |-- Validation data: ", n_data - idx_train)
    print(" |-- Test data: not yet created")

