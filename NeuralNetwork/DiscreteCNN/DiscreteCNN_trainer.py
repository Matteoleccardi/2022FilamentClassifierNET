'''
    Train routines, visualisation, and class
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from DiscreteCNN import DiscreteCNN
from DiscreteCNN_data_loader import DiscreteCNN_Dataset

if __name__ == "__main__":
    # Directory check
    if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
        print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
        quit()
    # Set hyperparams
    batch_size = 32
    num_workers = int(cpu_count()) if not torch.cuda.is_available() else 16
    lr0 = 0.005
    # Get training data
    csv_file = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data", "train.csv")
    train_dataset = DiscreteCNN_Dataset(csv_file, n_classes=5)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    # Get validation data
    csv_file = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data", "validation.csv")
    valid_dataset = DiscreteCNN_Dataset(csv_file, n_classes=5)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    # Get the used device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Create network
    model = DiscreteCNN(n_voxels_per_side=41, out_point_classes=5)
    model.to(device)
    # Loss, optmiser, lerning-rate trajectory
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9)

    # Training + Validation cycle
    plt.ion()
    fig = plt.figure()
    ax_t = fig.add_subplot()
    t_loss = []
    for epoch in range(15):
        # Training
        running_loss_t = 0.0
        for i, data in enumerate(train_loader, 0):
            # data
            inputs = data["input_t"].to(device)
            labels = data["label_t"].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # -------------
            running_loss_t += loss.item()
            t_loss.append(loss.item())
            ax_t.clear()
            ax_t.plot(range(0,len(t_loss)), t_loss)
            plt.pause(0.2)
        #t_loss.append(running_loss_t)
        # Validation
        A=1
        # Display images
        #ax_t.plot(range(1,epoch+1), t_loss)
        #plt.sleep(0.1)
    plt.show()
    print('Finished Training')