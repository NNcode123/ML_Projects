import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms
from Models.model import CNN_CAT_DOG
from PIL import Image
from utils.metrics import calc_accuracy
from utils.device import device
from pathlib import Path
import argparse
import os
import numpy
import pandas 
import torch.optim as optim



transform  = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

train_data = datasets.ImageFolder(root = "data\\PetImages", transform = transform)


train_data, val_data = data.random_split(train_data, [int(len(train_data)*0.8), len(train_data)-int(len(train_data)*0.8)])


def main():

    def train(EPOCH, LR, Batch):

        model = CNN_CAT_DOG().to(device)


        train_loader = data.DataLoader(dataset = train_data, batch_size = Batch, shuffle = True, num_workers = 0)
        

        optimizer = optim.Adam(params = model.parameters(), lr = LR )

        criterion = nn.CrossEntropyLoss()
        


        for epoch in range(EPOCH):

            total_loss, total_acc, sample_tot = 0,0, 0

            model.train()
            
            for (image, label) in enumerate(train_loader):

                image, label = image.to(device), label.to(device)

                output = model(image)

                loss = criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                accuracy = calc_accuracy(output, label)

                total_acc += accuracy

                sample_tot += label.numel()

                total_loss += loss.item()

            print(f"Epoch: {epoch}, train_accuracy {total_acc/len(train_loader):.4f}, train_Loss: {total_loss/len(train_loader)}")
        
        


        



    train(10,0.01, 32)






















if __name__ == "__main__":
    main()