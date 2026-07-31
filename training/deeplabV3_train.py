from training.unet_train import test_data,train_data
from utils.metrics import calc_accuracy
from utils.train_funcs import get_training_info, train_seg_model, optimizer_map, criterion_map
from utils.parent_dir import parent_dir
from utils.device import device
from utils.parser import Parser
import torch, torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import os
from Models.deeplabV3_ResNet50 import DeepLabV3



def main():

    config = get_training_info("deeplabv3")
    parser = Parser(
        epochs=config["training"]["epochs"],
        start_epoch=config["training"]["start_epoch"],
        lr=config["optimizer"]["params"]["lr"],
        momentum=config["optimizer"]["params"]["momentum"],
        weight_decay=config["optimizer"]["params"]["weight_decay"],
        batch_size=config["training"]["batch_size"],
    )
    args = parser.parse_args()

    model = DeepLabV3()
    optimizer_name = config["optimizer"]["type"]
    criterion_name = config["loss"]["type"]

    optimizer_cls = optimizer_map()[optimizer_name]
    criterion_cls = criterion_map()[criterion_name]
    
    optimizer = optimizer_cls(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    )
    criterion = criterion_cls()

    train_seg_model(model = model, train_dataset = train_data, optimizer = optimizer, criterion = criterion, batch_size = args.batch_size, epochs = args.epochs, 
                    checkpoint_dir = parent_dir/"checkpoint"/"DeepLabV3", device = device, start_epoch = args.start_epoch )



if __name__  == "__main__":

    main()
