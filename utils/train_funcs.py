
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from pathlib import Path 
from utils.metrics import calc_accuracy 
from utils.parent_dir import parent_dir
from utils.device import device

def train_results(BATCH, LR, EPOCHS, model, dataset, batch_prep_func, save_path):

        # Initialize the model, loss function, optimizer, and data transform
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        # Load the MNIST training dataset from the local data folder

        train_loader = DataLoader(dataset = dataset, batch_size= BATCH, shuffle = True, num_workers = 0)
        
        # Training loop over epochs and batches
        for epoch in range(EPOCHS):

            model.train()
            
            train_loss, train_acc = 0, 0


            for (image,label) in train_loader:
                
                image, label = image.to(device), label.to(device)

                image = batch_prep_func(image) if batch_prep_func else image

                output = model(image)

                loss= criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_acc += calc_accuracy(output, label)

                train_loss += loss.item()

                

            
            
            print(f"Epoch: {epoch}, train_accuracy: {train_acc/len(train_loader):.4f}, train_Loss: {train_loss/len(train_loader)}")

    

            Path.touch(save_path, exist_ok = True)

            torch.save({"model": model.state_dict(),
                        "optim": optimizer.state_dict()}, save_path)

            


        
        