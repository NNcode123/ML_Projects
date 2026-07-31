from Models.model import UNET
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.train_funcs import get_training_info, train_seg_model, optimizer_map, criterion_map
from utils.parent_dir import parent_dir
from utils.device import device
from utils.parser import Parser
import os



image_transform = v2.Compose(
    [v2.ToImage(),
    v2.Resize((256,256)),
    v2.ToDtype(
       torch.float32, scale = True
    )]
)

mask_transform = v2.Compose( [
    v2.Resize((256,256)),
    v2.ToImage(),
  
   
]
)

transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor()]
)

target_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()]
)

# Download and prepare the Oxford-IIIT Pet dataset for segmentation.
train_data = datasets.OxfordIIITPet(root = "data", target_types = "segmentation", transform = image_transform, target_transform = mask_transform,
                                     download = True )


test_data = datasets.OxfordIIITPet(root = "data", split = "test", target_types = "segmentation", transform = transform, target_transform = mask_transform,
                                   
                                    download = True)






def main():
    config = get_training_info("unet")
    parser = Parser(
        lr=config["optimizer"]["params"]["lr"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        start_epoch=config["training"]["start_epoch"],
    )
    args = parser.parse_args()

    model = UNET()
    optimizer_name = config["optimizer"]["type"]
    criterion_name = config["loss"]["type"]

    optimizer_cls = optimizer_map()[optimizer_name]
    criterion_cls = criterion_map()[criterion_name]

    optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    criterion = criterion_cls()
    train_seg_model(model = model, train_dataset = train_data, optimizer = optimizer, criterion = criterion, batch_size = args.batch_size, epochs = args.epochs, 
                    checkpoint_dir = parent_dir/"checkpoint"/"unet", device = device, start_epoch = args.start_epoch )


if __name__ == "__main__":
    main()
