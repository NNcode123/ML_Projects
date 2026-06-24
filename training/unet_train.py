from Models.model import UNET
import torch.nn as nn
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from utils.parent_dir import parent_dir
import os
from pathlib import Path
from utils.metrics import calc_accuracy
from torchinfo import summary



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

test_data = datasets.OxfordIIITPet(root = "data", split = "test", target_types = "segmentation", transform = transform, target_transform = transform,
                                   
                                    download = True)







def main():



    def train(BATCH, LR, EPOCH, start_epoch = -1):

        # Initialize U-Net model and optimizer
        model = UNET().to(torch.device('cuda'))
        optimizer = torch.optim.Adam(params =model.parameters(), lr = LR )

        if (start_epoch != -1):

            try:
                fl_name = start_epoch - 1
                checkpoint = torch.load( Path("checkpoint") / "unet" / f"epoch_{fl_name}.pth" )

                # Resume training from a saved checkpoint if requested
                model.load_state_dict(checkpoint["model"])

                optimizer.load_state_dict(checkpoint["optim"])
            except:
                raise RuntimeError(f"Model has not been trained up to {start_epoch} epochs ")

       
        
        train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH, num_workers = 0, shuffle = True)

        

        criterion = nn.CrossEntropyLoss()

        model.train()

        epochs = range(start_epoch, EPOCH + start_epoch) if start_epoch != -1 else range(EPOCH)

        # Loop over training epochs and compute per-batch loss/accuracy
        for epoch in epochs:
            tot_loss, tot_acc = 0, 0

            for image, mask in train_data_loader:
                image,mask = image.to(torch.device('cuda')), mask.to(torch.device('cuda'))

                output = model(image)

                optimizer.zero_grad()

                loss = criterion(output, mask.squeeze().long() - 1)

                loss.backward()

                optimizer.step()

                tot_loss += loss.item()

                tot_acc += calc_accuracy(output, mask.squeeze().long()-1)

            avg_acc = tot_acc/len(train_data_loader)
            avg_loss = tot_loss/len(train_data_loader)
            mod_path: Path = parent_dir /"checkpoint"/ "unet"
            fil_path = mod_path/ ("epoch_" + str(epoch)+".pth")
            mod_path.mkdir(exist_ok = True)
            fil_path.touch(exist_ok =  True)
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "accuracy": avg_acc,
                        "total_loss": avg_loss

            }, fil_path)

            print(f"Epoch: {epoch} Train_accuracy {avg_acc}, Train_Loss: {avg_loss}")


       
    train(64, 0.002, 15, 60)


if __name__ == "__main__":
    main()
