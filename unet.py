from Models.model import UNET
import torch.nn as nn
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
from pathlib import Path
from cnn_train import calc_accuracy
from torchinfo import summary

y = torch.randn(64,3,576,576)
model = UNET()
summary(model, input_size = (1,3,576,576))

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
   # v2.Lambda()
   
]
)

transform = transforms.Compose(
    #[transforms.Lambda(lambda x: x.convert("RGB")),
    [transforms.Resize((256,256)),
    transforms.ToTensor()]
)

target_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()]
)

train_data = datasets.OxfordIIITPet(root = "data", target_types = "segmentation", transform = image_transform, target_transform = mask_transform,
                                     download = True )

test_data = datasets.OxfordIIITPet(root = "data", split = "test", target_types = "segmentation", transform = transform, target_transform = transform,
                                   
                                    download = True)

print(type(train_data), type(test_data))






def main():



    def train(BATCH, LR, EPOCH):

        model = UNET().to(torch.device('cuda'))
        
        train_data_loader = DataLoader(dataset = train_data, batch_size = BATCH, num_workers = 0, shuffle = True)

        optimizer = torch.optim.Adam(params =model.parameters(), lr = LR )

        criterion = nn.CrossEntropyLoss()

        model.train()

        for epoch in range(EPOCH):

            tot_loss, tot_acc = 0, 0

            for ind, (image, mask) in enumerate(train_data_loader):
                image,mask = image.to(torch.device('cuda')), mask.to(torch.device('cuda'))

                output = model(image)


            
                """
                if (ind <= 5):
                    mask_sec  = mask[:,:10,:10]
                    output_sec = output[...,:10,:10]
                    print(f"mask: {mask_sec} \n\n")
                    print(f"mask_shape: {mask.shape}, output_shape: {output.argmax(axis = 1).shape}")
                    print(f"output: {output_sec.argmax(dim = 1)} \n\n")
                    print (f"mask_uq: {mask_sec.unique()} \n\n")
                """
                
                

                optimizer.zero_grad()

                loss = criterion(output, mask.squeeze().long() - 1)

                loss.backward()

                optimizer.step()

                
                


                tot_loss += loss.item()

                tot_acc += calc_accuracy(output, mask.squeeze().long()-1)

            print(f"Epoch: {epoch} Val_accuracy {tot_acc/len(train_data_loader):.4f}, Val_Loss: {tot_loss/len(train_data_loader):.4f}")

        #model_path =  
    train(64, 0.002, 30)


if __name__ == "__main__":
    main()
