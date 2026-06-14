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
    


def grad_cam(image: torch.Tensor, model: nn.Module)-> torch.Tensor:
    image = image.unsqueeze(0).to('cuda')

    features: torch.Tensor = model.features(image)

    features.retain_grad()

    output = model.classifier(model.linear(features))

    predicted_logit = output.argmax(axis = 1)
    
    output[:,predicted_logit].backward()

    with torch.no_grad():
        activation: torch.Tensor = features.grad.squeeze().mean(dim = (1,2))
        print(f"features_shape: {features.shape}, alpha_c: {activation.shape}")
        return torch.sum(features.squeeze()*activation.reshape(activation.shape[0],1,1), axis = 0)


import torch.nn.functional as F
import matplotlib.pyplot as plt

def overlay(heatmap: torch.Tensor, image: torch.Tensor, alpha: float = 0.7):

   
    image = image.squeeze().cpu()
    heatmap = heatmap.cpu()

   
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    heatmap = F.interpolate(heatmap, size=image.shape[1:], mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze()

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())


    image = image.permute(1, 2, 0)

   
    image = (image - image.min()) / (image.max() - image.min())


    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap.numpy())[..., :3]  # drop alpha channel

    heatmap_rgb = torch.tensor(heatmap_rgb)

    
    overlay = (1 - alpha) * image + alpha * heatmap_rgb

    overlay = overlay.clamp(0, 1)

    return overlay


    


fig_1, axs_1 = plt.subplots(nrows = 2, ncols = 8)

cat_features = feature_map(cat_image, model).squeeze()



print(cat_features.shape)



for ind in range(16):
    axs_1[ind//8, ind % 8].imshow(cat_features[ind,...].cpu())
    axs_1[ind//8, ind%8].axis("off")

fig_1.suptitle("Cat Feature Maps", fontsize = 16)
fig_1.savefig("Images\\features.png")




fig_2, axs_2 = plt.subplots(nrows = 1, ncols =2, figsize = (10,8))
axs_2[0].imshow(cat_image.permute(1,2,0))
axs_2[0].set_title("Original Image")
axs_2[0].axis("off")
axs_2[1].imshow(overlay(grad_cam(cat_image,model), cat_image))
axs_2[1].set_title("Heatmap Image")
axs_2[1].axis("off")

fig_2.savefig("Images\\heatmap_orig_comparison.png")

plt.show()









