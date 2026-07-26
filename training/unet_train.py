from Models.model import UNET
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.train_funcs import train_seg_model
from utils.parent_dir import parent_dir
from utils.device import device
import os
from pathlib import Path



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
    model = UNET()
    optimizer = optim.Adam(params = model.parameters(), lr = 0.99)
    train_seg_model(model = model, train_dataset = train_data, optimizer = optimizer, criterion = nn.CrossEntropyLoss(), batch_size = 64, epochs = 3, 
                    checkpoint_dir = parent_dir/"checkpoint"/"unet", device = device, start_epoch = 1 )


if __name__ == "__main__":
    main()
