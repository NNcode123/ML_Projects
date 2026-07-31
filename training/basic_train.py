
from utils.parent_dir import parent_dir
from utils.parser import Parser
from utils.train_funcs import train_results, get_training_info, optimizer_map, criterion_map
from torchvision import datasets, transforms
from Models.model import BASIC_MODEL
import torch.nn as nn


# Basic MNIST training script
# This module defines the training workflow for the BASIC_MODEL and saves
# checkpoint files for the trained model and optimizer.

def main():

    transform = transforms.ToTensor()

    MNIST_Train = datasets.MNIST(root = "data", train= True,  transform = transform)

    config = get_training_info("BASIC")
    parser = Parser(
        lr=config["optimizer"]["params"]["lr"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
    )
    args = parser.parse_args()

    model = BASIC_MODEL()

    optimizer_name = config["optimizer"]["type"]
    criterion_name = config["loss"]["type"]

    optimizer_cls = optimizer_map()[optimizer_name]
    criterion_cls = criterion_map()[criterion_name]

    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    criterion = criterion_cls()

    train_results(args.batch_size, args.lr, args.epochs, model, optimizer, criterion,
                  MNIST_Train, lambda x: x.view(x.shape[0], -1), parent_dir / "checkpoint" / "basic" / "saved_dnn_model.pth")










    








    

    

    
    




if __name__ == "__main__":
    main()


    

