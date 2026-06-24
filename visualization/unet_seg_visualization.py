import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from training.unet_train import train_data
import torch.utils.data as data
import torch
from Models.model import UNET




train_data_loader = data.DataLoader(dataset = train_data, batch_size = 64, num_workers = 0, shuffle = True)

image_batch, label = next(iter(train_data_loader))

image = image_batch[10, ...]

label = label[10, ...]

model = UNET().to(torch.device('cuda'))

checkpoint = torch.load( Path("checkpoint") / "unet" / "epoch_74.pth" )

model.load_state_dict(checkpoint["model"])



predicted: torch.Tensor = model(image.to(torch.device('cuda')).unsqueeze(0)).cpu().argmax(axis = 1)

print(predicted.shape)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20,7))

prediction = predicted.detach().cpu().permute(1,2,0)
ground_truth = label.permute(1,2,0)
diff = (prediction != ground_truth)

axs[0].imshow(prediction)
axs[1].imshow(ground_truth)
axs[2].imshow(diff)

plt.show()