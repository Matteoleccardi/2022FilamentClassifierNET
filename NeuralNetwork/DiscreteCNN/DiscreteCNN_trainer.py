'''
    Train routines, visualisation, and class
'''
import os
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from DiscreteCNN import DiscreteCNN_core
from DiscreteCNN_data_loader import DiscreteCNN_Dataset

if __name__ == "__main__":
    # Directory check
    if os.path.basename(os.getcwd()) != "2022FilamentClassifierNET":
        print("ERROR:\nEvery script must be executed from inside the 2022FilamentClassifierNET directory")
        quit()
    # ste model name
    now = datetime.now()
    model_name = now.strftime("model_%Y%m%d_%H%M%S.pt")
    # Set hyperparams
    n_classes = 5
    n_training_steps = 750
    batch_size = 32
    num_workers = np.min([batch_size, int(cpu_count()) if not torch.cuda.is_available() else torch.cuda.device_count()])
    lr0 = 0.001
    # Get training data
    csv_file = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data", "train.csv")
    train_dataset = DiscreteCNN_Dataset(csv_file, n_classes=n_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    # Get validation data
    csv_file = os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "train_valid_data", "validation.csv")
    valid_dataset = DiscreteCNN_Dataset(csv_file, n_classes=n_classes)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    # Get the used device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Create network
    model = DiscreteCNN_core(n_voxels_per_side=41, out_point_classes=n_classes)
    model.to(device)
    # Loss, optmiser, lerning-rate trajectory
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9)

    # Training + Validation cycle
    plt.ion()
    fig = plt.figure()
    spec = fig.add_gridspec(2, 3)
    ax_tv = fig.add_subplot(spec[:,:2]) # losses
    ax_im = fig.add_subplot(spec[0,2])  # confusion matrix
    ax_rw = fig.add_subplot(spec[1,2])  # accuracy chart
    t_loss_list = []
    v_loss_list, v_steps_list, v_right_wrong_list = [], [], []
    for training_step in range(n_training_steps):
        print(f"Train step {training_step:3d}", end="\r")
        # Training ------------------------------
        data = next(iter(train_loader))
        # unpack data
        inputs = data["input_t"].to(device)
        labels = data["label_t"].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        t_loss = criterion(outputs, labels)
        t_loss.backward()
        optimizer.step()
        # Learning rate
        if training_step < int(n_training_steps/5):
            lr = ((lr0*2-lr0)/int(n_training_steps/5))*training_step + lr0
        else:
            lr = -(2*lr0/(n_training_steps+1-int(n_training_steps/5)))*training_step + 3*lr0
        for g in optimizer.param_groups:
            g['lr'] = lr
        # save res
        t_loss_list.append(t_loss.item())
        # Validation ----------------------------
        if (training_step == 0) or (training_step%5 == 0):
            v_loss = 0.0
            confusion_matrix_image = np.zeros((n_classes, n_classes))
            right, wrong = 0, 0
            with torch.no_grad():
                for i, v_data in enumerate(valid_loader, 0):
                    inputs = v_data["input_t"].to(device)
                    labels = v_data["label_t"].to(device)
                    outputs = model(inputs)
                    # save res
                    v_loss += criterion(outputs, labels).item()/len(valid_loader)
                    for rowL, rowO in zip(labels, outputs):
                        idx_r = int( rowL.cpu().detach().numpy() )
                        idx_c = int( np.argmax( torch.nn.Softmax(dim=-1)(rowO).cpu().detach().numpy()) )
                        confusion_matrix_image[idx_r, idx_c] += 1
                        if idx_r == idx_c:
                            right += 1
                        else:
                            wrong += 1
            v_loss_list.append(v_loss)
            v_steps_list.append(training_step)
            v_right_wrong_list.append([right, wrong])
        # Display images -------------------------
        #
        ax_tv.clear()
        ax_tv.plot(t_loss_list, "r-", label="T", linewidth=0.7)
        ax_tv.plot(v_steps_list,v_loss_list, "b-", label="V", linewidth=1.4)
        ax_tv.set_ylabel("Cross Entropy loss")
        ax_tv.set_xlabel("Training steps")
        ax_tv.set_title("Learning trajectory")
        ax_tv.legend()
        ax_tv.grid()
        #
        for r in range(confusion_matrix_image.shape[0]):
            confusion_matrix_image[r,:] = confusion_matrix_image[r,:] / np.sum(confusion_matrix_image[r,:]) if np.sum(confusion_matrix_image[r,:]) != 0 else 0*confusion_matrix_image[r,:]
        ax_im.clear()
        ax_im.imshow(confusion_matrix_image, vmin=0, vmax=1, cmap="bone")
        ax_im.set_ylabel("True class (%)")
        ax_im.set_xlabel("Predicted class (%)")
        ax_im.set_title("Confusion matrix\n(with row-wise normalisation)")
        #
        rwl = np.array(v_right_wrong_list)
        n_valid = rwl[-1,0] + rwl[-1,1]
        ax_rw.plot(v_steps_list, rwl[:,0], label="Right classification")
        ax_rw.plot([0, v_steps_list[-1]], [n_valid, n_valid], "--", color="black", label="Tot")
        ax_rw.legend()
        ax_rw.set_title(f"Correct inferences: {rwl[-1,0]:4d} ({100 * rwl[-1,0]/n_valid:3.2f}%)")
        plt.pause(0.5)
        # Save model params if validation is the best
        if training_step != 0:
            if np.min(v_loss_list) == v_loss_list[-1]:
                torch.save(model.state_dict(), os.path.join(os.getcwd(), "NeuralNetwork", "DiscreteCNN", "trained_networks", model_name))
    print('Finished Training')
    plt.ioff()
    plt.show()
    
    