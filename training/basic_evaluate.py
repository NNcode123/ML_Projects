import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.device import device
from utils.parent_dir import parent_dir
from utils.metrics import calc_accuracy
from Models.model import BASIC_MODEL





def main():
    model = BASIC_MODEL()

    # Load the trained model weights from checkpoint
    model.load_state_dict(torch.load(parent_dir /"checkpoint" / "basic" / "dnn_model.pth"))

    model.to(device)

    BATCH  = 128

    transform = transforms.ToTensor()

    # Load the MNIST test split for evaluation
    MNIST_Test = datasets.MNIST(root = "data", train = False, transform = transform )

    test_data_loader = DataLoader(dataset = MNIST_Test, batch_size = BATCH//2, shuffle = True, num_workers = 2)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    # Run evaluation without gradient tracking for inference
    with torch.no_grad():

        loss_num, accuracy = 0, 0

        for (image,label) in test_data_loader:

            image, label = image.to(device), label.to(device)

            output = model(image.view(image.size(0), -1))

            loss = criterion(output,label)

            loss_num += loss.item()

            accuracy += calc_accuracy(output, label)

            

        print(f"Accuracy: {accuracy/len(test_data_loader)}, Loss: {loss_num/len(test_data_loader)}")



if __name__ == "__main__":
    main()