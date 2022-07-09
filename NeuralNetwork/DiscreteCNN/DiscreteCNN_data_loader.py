'''
    Dataset class to be used together with pytorch's dataloader
'''
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class DiscreteCNN_Dataset(Dataset):
    def __init__(self, csv_file, n_classes=5):
        """
        Input:
            csv_file (string): Absolute path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.csv = np.loadtxt(csv_file, delimiter=",", skiprows=1, dtype=[("fname","U256"),("class","i4")])
        self.length = int( self.csv.shape[0] )
        self.n_classes = n_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Input:
                idx: an int, a list or a monodimensional tensor
        '''
        # Clean input
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx).__module__ == np.__name__:
            idx = idx.tolist()
        if type(idx) is slice:
            idx = list(range(idx.stop)[idx])
        # Get data and labels
        if type(idx) is list:
            label_tensor = torch.zeros((len(idx), 1))
            im_list = []
            for i, c in zip(idx, range(len(idx)) ):
                im_list.append( torch.load(self.csv[i]["fname"]) )
                label_tensor[c, 0] = int(self.csv[i]["class"])
            im_tensor = torch.cat(im_list, dim=0) # torch.cat(tuple(im_list), dim=0)
        else:
            im_tensor = torch.load(self.csv[idx]["fname"])
            im_tensor = torch.squeeze(im_tensor, 0)
            label_tensor = torch.tensor( int(self.csv[idx]["class"]) )
        sample = {"input_t": im_tensor, "label_t": label_tensor}
        return sample


# Usage example
if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
        print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
        quit()
    csv_file = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data", "validation.csv")
    valid_dts = DiscreteCNN_Dataset(csv_file, n_classes=5)

    print(valid_dts[1]["input_t"].shape, valid_dts[1]["label_t"])
    print(valid_dts[3:8]["input_t"].shape, valid_dts[3:8]["label_t"])