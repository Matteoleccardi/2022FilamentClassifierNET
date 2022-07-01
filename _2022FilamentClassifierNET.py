
'''
    Main executable file - used while developing
'''

import numpy as np
import matplotlib.pyplot as plt
from Dataset.DatasetCreator_utils import *
import time



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
from NeuralNetwork.DiscreteCNN.DiscreteCNN import *
# model
n_voxels_per_side=41
model = DiscreteCNN(n_voxels_per_side=n_voxels_per_side, out_point_classes=5)
model.eval()
# filament
np.random.seed(0)
pts = makefilament_001(120, 10)
# tensor from filament
t0 = time.time()
vol = preprocess(pts[:,:3], index=15, n_voxels_per_side=n_voxels_per_side)
t1=time.time()
print( f"Elapsed time: {t1-t0:.6f} seconds."); quit()
##################à YOU ARE TESTING THE SPEED OF THIS CONVERSION
# OLD WAY:  2.1430
# NEW WAY: TO DO !#------------------------------------------------------     <---

outputs = model(vol)
print(outputs)
view_input_volume(torch.squeeze(torch.squeeze(vol,0),0))



quit()

