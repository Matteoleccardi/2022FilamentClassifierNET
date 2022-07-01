import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
from torch import nn


def preprocess(filament: np.ndarray, index: int, n_voxels_per_side: int=31) -> torch.Tensor:
    '''
        Input:
            filament: all the filaments points in the point cloud. A numpy.ndarray of dimension N x 3: | x | y | z |
            index: index of the point to be classified.
            n_voxels_per_side: number of voxels per side in the cubic voxelised volume. Default: 21 voxels.
        Output:
            volume: a torch.tensor voxel volume representing the nearest surrounding of the point under scrutiny
    '''
    # Input checks
    if index > filament.shape[0]-1:
        print(f"Invalid index {index} passed to function \"preprocess\", when filament is {filament.shape[0]} long. Quitting...")
        quit()
    #  Create 3D voxels volume (indexes: [z, y, x])
    if n_voxels_per_side % 2 == 0:
        n_voxels_per_side += 1
    volume = torch.zeros([n_voxels_per_side, n_voxels_per_side, n_voxels_per_side])
    if filament.shape[0] == 1:
        i = int( np.floor(n_voxels_per_side/2) + 1) - 1
        volume[i,i,i] = 1
    else:
        # Find distance to be used as pixel spacing in the volume voxelisation
        # -> Median distance between each point and its nn - useful when classifying isolated points
        d_list = 1e20 * np.ones((filament.shape[0],))
        for i in range(filament.shape[0]):
            d = np.linalg.norm(filament[i,:] - filament, axis=1)
            d = np.partition(d, kth=2)[1]
            d_list[i] = d
        d_median = np.median(d_list)
        # -> Distance between the considered point to be classified and its nn
        d_nn = np.linalg.norm(filament[index,:] - filament, axis=1)
        d_nn = np.partition(d_nn, kth=2)[1] / np.sqrt(2)
        # -> Find minimum distance and use that
        d_min = np.min([d_median, d_nn])
        # Populate volume
        [cx, cy, cz] = filament[index,:]
        half_side = d_min/2 + np.floor(n_voxels_per_side/2)*d_min
        x_lowlim, x_highlim = cx - half_side, cx + half_side
        y_lowlim, y_highlim = cy - half_side, cy + half_side
        z_lowlim, z_highlim = cz - half_side, cz + half_side
        x_centers = np.linspace(x_lowlim+d_min/2, x_highlim-d_min/2, num=n_voxels_per_side)
        y_centers = np.linspace(y_lowlim+d_min/2, y_highlim-d_min/2, num=n_voxels_per_side)
        z_centers = np.linspace(z_lowlim+d_min/2, z_highlim-d_min/2, num=n_voxels_per_side)
        cond = filament[:,0] >= x_lowlim ## ------------------ PROVA A VELOCIZZARE COSI
        cond = np.logical_and(cond, filament[:,0] <= x_highlim)
        cond = np.logical_and(cond, filament[:,1] >= y_lowlim)
        cond = np.logical_and(cond, filament[:,1] <= y_highlim)
        cond = np.logical_and(cond, filament[:,2] >= z_lowlim)
        cond = np.logical_and(cond, filament[:,2] <= z_highlim)
        for p in filament[cond,:]:
            xi = np.argmin( np.abs(x_centers - p[0]) )
            yi = np.argmin( np.abs(y_centers - p[1]) )
            zi = np.argmin( np.abs(z_centers - p[2]) )
            if isinstance(xi, np.ndarray): xi = xi[0]
            if isinstance(yi, np.ndarray): yi = yi[0]
            if isinstance(zi, np.ndarray): zi = zi[0]
            volume[zi, yi, xi] += 1
    # Set max voxel value to one
    if torch.max(volume) != 0:
        volume = volume / torch.max(volume)
    # Output
    volume = torch.unsqueeze(volume, 0)
    volume = torch.unsqueeze(volume, 0)
    return volume


class DiscreteCNN(nn.Module):
    def __init__(self, n_voxels_per_side, out_point_classes):
        super(DiscreteCNN, self).__init__()
        self.n_voxels_per_side = n_voxels_per_side
        self.conv1 = self.ConvLayer3D(1, 32)
        self.conv2 = self.ConvLayer3D(32, 64)
        self.conv3 = self.ConvLayer3D(64, 16, kernel_size=5)
        self.fc1   = self.FCLayer(574992, 64)
        self.fc2  = nn.Linear(64, out_point_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, volume):
        '''
            Input
            -----
                volume: a torch 5-dimensional tensor: [batch_idx, channel_idx, z_idx, y_idx, x_idx] 
            Output
            ------
                Probability distribution over "out_point_classes"
        '''
        # NN stack
        out = self.conv1( volume )
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        print(out.shape, ((self.n_voxels_per_side-3*3)**3)*16)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        out = out.flatten()
        return out

    def ConvLayer3D(self, in_c, out_c, kernel_size=3):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
        return conv_layer

    def FCLayer(self, in_c, out_c):
        fc_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_c),
            nn.Dropout(p=0.10) 
        )
        return fc_layer
    

# define dataset creator (all images - class couples) in a subfolder

# define the dataloader in new file

# build train-validation cycles in new file -> save model parameters 



# Visualisation
def view_input_volume(volume: torch.Tensor):
    # Init img
    idx0 = int(volume.shape[0]/2 + 1)
    fig, ax = plt.subplots()
    im = ax.imshow(volume[idx0,:,:], cmap="gray")
    # Slider
    ax_slider = fig.add_axes([0.25, 0.05, 0.6, 0.02])
    slider = Slider(ax_slider, "Z-axis index", valmin=0, valmax=int(volume.shape[0]-1), valinit=idx0, valstep=1, valfmt='%d')
    global view_input_volume_data
    view_input_volume_data = [fig, ax, im, volume]
    slider.on_changed(update_view_input_volume)
    # Show image
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def update_view_input_volume(idx):
    global view_input_volume_data
    view_input_volume_data[2].set_data((view_input_volume_data[3])[int(idx),:,:])
    view_input_volume_data[1].set_title(f"Input volume plotted for $Z_idx$ = {idx}")
    view_input_volume_data[0].canvas.draw_idle()

