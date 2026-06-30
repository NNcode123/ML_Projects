from training.unet_train import test_data,train_data
from torch.nn import CrossEntropyLoss as criterion 
from utils.metrics import calc_accuracy
from utils.parent_dir import parent_dir
from utils.device import device
import torch, torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import os
from Models.deeplabV3_ResNet50 import DeepLabV3



def main():


    def deeplabv3_train(EPOCH, LR, BATCH):

        model = DeepLabV3().to(device)
        model.train()
        optimizer = optim.SGD(params = model.parameters(), lr = LR, momentum = 0.9, weight_decay = 1e-4 )

        loader = data.DataLoader(dataset = train_data, batch_size = BATCH, shuffle = True, num_workers = 4)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCH):

            epoch_acc_tot, epoch_loss_tot = 0,0

            for (image,label) in loader:
                image, label = image.to(device), label.to(device)

                label = label.squeeze().long() - 1

                output = model(image)

                optimizer.zero_grad()

                loss = criterion(output, label)

                epoch_loss_tot += loss.item()

                epoch_acc_tot += calc_accuracy(output, label)

                loss.backward()

                optimizer.step()

            avg_acc = epoch_acc_tot/len(loader)
            avg_loss = epoch_loss_tot/len(loader)

            print(f"Epoch: {epoch} Train_accuracy {avg_acc}, Train_Loss: {avg_loss}")

    deeplabv3_train(13, 0.001, 4)























if __name__  == "__main__":

    main()
