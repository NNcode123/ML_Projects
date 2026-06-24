
import argparse as arg
from utils.parent_dir import parent_dir
from utils.train_funcs import train_results
from torchvision import datasets, transforms
from pathlib import Path
from Models.model import BASIC_MODEL


# Basic MNIST training script
# This module defines the training workflow for the BASIC_MODEL and saves
# checkpoint files for the trained model and optimizer.

def main():

    transform = transforms.ToTensor()

    MNIST_Train = datasets.MNIST(root = "data", train= True,  transform = transform)

    # Parse command-line arguments for learning rate, batch size, and epochs
    arg_collector = arg.ArgumentParser()

    arg_collector.add_argument(  "--lr", type = float)
    arg_collector.add_argument("--batch_size", type = int)
    arg_collector.add_argument("--epochs", type = int)

    args = arg_collector.parse_args()

    train_results(args.batch_size, args.lr, args.epochs, BASIC_MODEL(), 
                  MNIST_Train, lambda x: x.view(x.shape[0], -1), parent_dir / "checkpoint" / "basic" / "saved_dnn_model.pth")










    








    

    

    
    




if __name__ == "__main__":
    main()


    

