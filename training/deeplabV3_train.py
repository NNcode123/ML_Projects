from training.unet_train import test_data,train_data
from torch.nn import CrossEntropyLoss as criterion 
from utils.metrics import calc_accuracy
from utils.train_funcs import train_seg_model
from utils.parent_dir import parent_dir
from utils.device import device
import torch, torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import os
from Models.deeplabV3_ResNet50 import DeepLabV3



def main():

    model = DeepLabV3()
    
    optimizer = optim.SGD(
    model.parameters(),
    lr=0.007,
    momentum=0.9,
    weight_decay=1e-4,
    )

    train_seg_model(model = model, train_dataset = train_data, optimizer = optimizer, criterion = nn.CrossEntropyLoss(), batch_size = 4, epochs = 34, 
                    checkpoint_dir = parent_dir/"checkpoint"/"DeepLabV3", device = device, start_epoch = 1 )



if __name__  == "__main__":

    main()
