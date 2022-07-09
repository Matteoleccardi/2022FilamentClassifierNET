
'''
    Main executable file - used while developing
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from Dataset.DatasetCreator_utils import *

# Manage directories
if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
    print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
    quit()
    


# Example filament
if 0:
    p=makefilament_classes_demo()
    visualize_dataset_3D(p)

# fil creation trial
if 0:
    np.random.seed(0)
    for i in range(1):
        n = int( np.random.triangular(50, 100, 300) )
        nb = int( np.random.choice([1,2,3]) )
        l = np.random.uniform(0.01, 1000)
        n_fil = int( np.random.triangular(1, 5, 15) )
        #pts = makefilament_006(n_points=n, n_branching_points=nb)#, n_fil)
        pts = makefilament_aggregator(makefilament_003, n, l, 10)
        visualize_dataset_3D(pts)

# NN models tests
if 1:
    from NeuralNetwork.DiscreteCNN.DiscreteCNN import DiscreteCNN
    # filament
    pts=makefilament_classes_demo()

    # model
    model_name = "model_20220708_181634.pt"
    n_voxels_per_side=41
    model = DiscreteCNN(n_voxels_per_side=n_voxels_per_side, out_point_classes=5)
    model.load_state_dict(torch.load("./NeuralNetwork/DiscreteCNN/trained_networks/"+model_name))
    model.eval()

    # try classification
    inferred_classes = np.zeros((pts.shape[0],1))
    for index in range(pts.shape[0]):
        print(f"Classified {index+1} of {pts.shape[0]}", end="\r")
        with torch.no_grad():
            outputs = model(pts[:,:3], index)
        cl = np.argmax(outputs.numpy())
        inferred_classes[index,0] = cl

    # view output
    visualize_dataset_3D(pts)
    out = np.append(pts[:,:3],inferred_classes, axis=1)
    out = np.append(out, np.zeros((pts.shape[0],1)), axis=1)
    visualize_dataset_3D(out)






