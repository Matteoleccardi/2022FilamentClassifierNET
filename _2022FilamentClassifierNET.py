
'''

'''

import numpy as np
import matplotlib.pyplot as plt
from Dataset.DatasetCreator_utils import *



# Example filament
if 0:
    p=make_classes_demo()
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

# NN models
from NeuralNetwork.DiscreteCNN import *
n_voxels_per_side=41
model = DiscreteCNN(n_voxels_per_side=n_voxels_per_side, out_point_classes=5)
model.eval()
pts = makefilament_001(120, 10)
vol = preprocess(pts[:,:3], index=15, n_voxels_per_side=n_voxels_per_side)
outputs = model(vol)
print(outputs)
view_input_volume(torch.squeeze(torch.squeeze(vol,0),0))



quit()

