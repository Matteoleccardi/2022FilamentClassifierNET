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
    # Find minimum distance between the point of interest and any other point in the filament
    dists = np.linalg.norm(filament[index,:] - filament, axis=1)
    dists[dists == 0] = 1e20
    d_min = np.min(dists)/np.sqrt(2)
    # Create 3D voxels volume (indexes: [z, y, x])
    if n_voxels_per_side % 2 == 0: n_voxels_per_side += 1
    volume = torch.zeros([n_voxels_per_side, n_voxels_per_side, n_voxels_per_side])
    # Populate volume
    [cx, cy, cz] = filament[index,:]
    x_lowlim = cx - np.floor(n_voxels_per_side/2)*d_min - d_min/2
    y_lowlim = cy - np.floor(n_voxels_per_side/2)*d_min - d_min/2
    z_lowlim = cz - np.floor(n_voxels_per_side/2)*d_min - d_min/2
    for xi in range(n_voxels_per_side):
        xl, xh = x_lowlim + xi*d_min, x_lowlim + (xi + 1)*d_min
        for yi in range(n_voxels_per_side):
            yl, yh = y_lowlim + yi*d_min, y_lowlim + (yi + 1)*d_min
            for zi in range(n_voxels_per_side):
                zl, zh = z_lowlim + zi*d_min, z_lowlim + (zi + 1)*d_min
                # Populate voxel
                for fp in filament:
                    if ((xl <= fp[0]) and (fp[0] < xh)) and ((yl <= fp[1]) and (fp[1] < yh)) and ((zl <= fp[2]) and (fp[2] < zh)):
                        volume[-1-zi, -1-yi, xi] += 1
    # Pseudo-normalisation
    if torch.max(volume) != 0:
        volume = volume / torch.max(volume)
    # Output
    volume = torch.unsqueeze(torch.unsqueeze(volume,0), 0)
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

