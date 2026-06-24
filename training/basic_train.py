
import torch.optim
import random
import os 
import argparse as arg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.parent_dir import parent_dir
from utils.metrics import calc_accuracy
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from Models.model import BASIC_MODEL
from Models.model import BASIC_CNN

# Basic MNIST training script
# This module defines the training workflow for the BASIC_MODEL and saves
# checkpoint files for the trained model and optimizer.

def main():


    


    # Select GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)



    BATCH = 64
    LR = 0.005
    EPOCHS = 150



    def train_results(BATCH, LR, EPOCHS):

        # Initialize the model, loss function, optimizer, and data transform
        model = BASIC_MODEL().to(torch.device(device))

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        transform = transforms.ToTensor()

        # Load the MNIST training dataset from the local data folder
        MNIST_Train = datasets.MNIST(root = "data", train= True,  transform = transform)

        


        train_data_loader = DataLoader(dataset = MNIST_Train, batch_size= BATCH, shuffle = True, num_workers = 0)

        


        

        print(f"num_parameters: {sum(p.numel() for p in model.parameters())}")



        

        # Training loop over epochs and batches
        for epoch in range(EPOCHS):

            model.train()
            
            train_loss, train_acc = 0, 0


            for (image,label) in train_data_loader:
                
            


                image, label = image.to(device), label.to(device)

                output = model(image.view(image.shape[0],-1))

                loss= criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                train_acc += calc_accuracy(output, label)

                train_loss += loss.item()

                optimizer.step()


            
            
            print(f"Epoch: {epoch}, Loss: {train_loss/len(train_data_loader):.4f}")

        # Save model and optimizer checkpoints for later evaluation or resuming
        model_path = parent_dir/ "checkpoint" / "basic" /"dnn_model.pth"

        optimizer_path = parent_dir /"checkpoint" /"basic"/"dnn_optimizer.pth"

        Path.touch(model_path)

        Path.touch(optimizer_path)

        torch.save(model.state_dict(), model_path)

        torch.save(optimizer.state_dict(), optimizer_path)  


    

    # Parse command-line arguments for learning rate, batch size, and epochs
    arg_collector = arg.ArgumentParser()

    arg_collector.add_argument(  "--lr", type = float)
    arg_collector.add_argument("--batch_size", type = int)
    arg_collector.add_argument("--epochs", type = int)


    args = arg_collector.parse_args()

    train_results(args.batch_size, args.lr, args.epochs)








    








    

    

    
    




if __name__ == "__main__":
    main()


    

