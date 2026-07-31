import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
from torchvision import datasets, transforms
from Models.model import CNN_CAT_DOG
from PIL import Image
import PIL
from utils.metrics import calc_accuracy
from utils.device import device
from utils.parent_dir import parent_dir
from utils.parser import Parser
from utils.train_funcs import get_training_info, train_results, optimizer_map, criterion_map

import os

import torch.optim as optim

# CNN training script for cat vs dog classification using a small custom network.
# This script defines preprocessing, dataset splitting, and the training loop.

transform  = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])


#check for corrupt images in the data/PetImages directory and remove them accordingly
for file_name in (parent_dir / "PetImages").rglob("*.jpg"):

    try:
        image = Image.open(file_name)
        image.verify()
    
    except PIL.UnidentifiedImageError:
        print(f"{file_name} is corrupted, and will be removed accordingly.")
        os.remove(file_name)



train_data = datasets.ImageFolder(root = "data\\PetImages", transform = transform)


train_data, val_data = data.random_split(train_data, [int(len(train_data)*0.8), len(train_data)-int(len(train_data)*0.8)])


def main():

    config = get_training_info("CNN_CAT_DOG")
    parser = Parser(
        lr=config["optimizer"]["params"]["lr"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
    )
    args = parser.parse_args()

    model = CNN_CAT_DOG()
    optimizer_name = config["optimizer"]["type"]
    criterion_name = config["loss"]["type"]

    optimizer_cls = optimizer_map()[optimizer_name]
    criterion_cls = criterion_map()[criterion_name]

    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    criterion = criterion_cls()

    train_results(args.batch_size, args.lr, args.epochs, model, optimizer, criterion, train_data, None, parent_dir / "checkpoint" / "cnn" / "saved_cnn_model.pth")

    


if __name__ == "__main__":
    main()