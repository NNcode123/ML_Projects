import matplotlib.pyplot as plt
from Models.model import CNN_CAT_DOG
import torch.nn as nn
import torch
import random
from cnn_train import transform
from PIL import Image 
from pathlib import Path
import os 
import matplotlib.pyplot as plt


model = CNN_CAT_DOG().to(torch.device('cuda'))

model.load_state_dict(torch.load(os.path.join("checkpoint","cnn","model_epoch_9.pth")))


print(f"Total_params:sum(param.numel() for param in model.parameters())")


cat_image, dog_image = transform(Image.open(os.path.join("data","PetImages","Cat",str(random.randint(0,50))+".jpg")).convert("RGB")) ,        transform(Image.open(os.path.join("data","PetImages","Cat",str(random.randint(0,50))+".jpg"))
            .convert("RGB"))



def feature_map(image: torch.tensor, model: nn.Module)->torch.Tensor:
    image = image.unsqueeze(0).to('cuda')
    with torch.no_grad():
        map = image
        for _ in range(8):
            map = model.features[_](map)
        return map
    


fig, axs = plt.subplots(nrows = 2, ncols = 8)

cat_features = feature_map(cat_image, model).squeeze()

print(cat_features.shape)

for ind in range(16):
    axs[ind//8, ind % 8].imshow(cat_features[ind,...].cpu())
    fig.tight_layout()

plt.show()






