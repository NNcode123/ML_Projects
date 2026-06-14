import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms
from Models.model import CNN_CAT_DOG
from PIL import Image
from pathlib import Path
import argparse
import os
import numpy
import pandas 
import torch.optim as optim

def calc_accuracy(y_pred_matrix: torch.Tensor, actual_y: torch.Tensor) -> int:
    y_pred_class = y_pred_matrix.argmax(axis = 1) 

    return (y_pred_class == actual_y).sum().item()/actual_y.numel()

transform  = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

train_data = datasets.ImageFolder(root = "data\\PetImages", transform = transform)

print(train_data.class_to_idx)

train_data, val_data = data.random_split(train_data, [int(len(train_data)*0.8), len(train_data)-int(len(train_data)*0.8)])


def main():

   
              

   

    train_data = datasets.ImageFolder(root = "data\\PetImages", transform = transform)

    print(train_data.class_to_idx)

    train_data, val_data = data.random_split(train_data, [int(len(train_data)*0.8), len(train_data)-int(len(train_data)*0.8)])

    

    val_loader = data.DataLoader(dataset = val_data, batch_size = Batch, shuffle = True, num_workers = 0  )

    def train_validation(EPOCH, LR, Batch):

        model = CNN_CAT_DOG().to(torch.device('cuda'))


        train_loader = data.DataLoader(dataset = train_data, batch_size = Batch, shuffle = True, num_workers = 0)
        

        optimizer = optim.Adam(params = model.parameters(), lr = LR )

        criterion = nn.CrossEntropyLoss()

       



            



        

        


        for epoch in range(EPOCH):

            total_loss, total_acc, sample_tot = 0,0, 0

            model.train()
            
            for ind,(image, label) in enumerate(train_loader):

                image, label = image.to('cuda'), label.to('cuda')

                output = model(image)


                """
                if (ind <= 3):
                    print(f"Output:\n {output}\n\n")
                    print(f"Predicted: \n\n {output.detach().argmax(axis = 1)} \n\n")
                """

                loss = criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                accuracy = calc_accuracy(output, label)

                total_acc += accuracy

                sample_tot += label.numel()

                total_loss += loss.item()

                #print(f"Accuracy: {accuracy/ label.numel()}")

            path: Path = Path("checkpoint") / "cnn"  #/("model_epoch_"+str(epoch)+".pth"))
            path.mkdir(parents = True, exist_ok = True)
            path_1 = Path("checkpoint")/"cnn" #/("optim_epoch_"+str(epoch)+".pth"))
            path_1.mkdir(parents = True, exist_ok = True)
            path = path / ("model_epoch_"+str(epoch)+".pth")
            path_1 = path_1 /("optim_epoch_"+str(epoch)+".pth")
            path.touch(exist_ok = True)
            path_1.touch(exist_ok = True)
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(),path_1)


                
                
            print(f"Total_accuracy {total_acc/len(train_loader):.4f}, Loss: {total_loss/len(train_loader)}")

        



    train_validation(10,0.01, 32)






















if __name__ == "__main__":
    main()