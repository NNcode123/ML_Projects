from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from cnn_train import calc_accuracy
from cnn_train import val_data
from Models.model import CNN_CAT_DOG
import os


def main():

    model = CNN_CAT_DOG().to(torch.device('cuda'))

    model.load_state_dict(torch.load(os.path.join("checkpoint","cnn","model_epoch_9.pth")))

    val_loader = DataLoader(dataset = val_data, batch_size = 64, shuffle  = True, num_workers = 0)

    criterion = nn.CrossEntropyLoss()
                          
    model.eval()

    with torch.no_grad():
    
        total_loss, total_acc = 0,0
        
        for (image, label) in val_loader: 
        
            image, label = image.to('cuda'), label.to('cuda')

            output = model(image)

            loss = criterion(output, label)

            total_loss += loss.item()

            total_acc += calc_accuracy(output, label)

        
        print(f"Loss: {total_loss/len(val_loader)}, Accuracy: {total_acc/len(val_loader)}")





if __name__ == "__main__":
    main()
    



