import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from utils.parent_dir import parent_dir
from training.unet_train import train_data
import utils.metrics as metr
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

output = model(image.to(torch.device('cuda')).unsqueeze(0)).cpu()

predicted =  output.argmax(axis = 1)


u = label.clone()

print(f"\n test io score:{metr.iou_score(u, label, 1)} dice score: {metr.dice_score(u, label, 1)} \n\n")


print(f"\n iou score:{metr.miou_score(output, label)} dice score: {metr.mdice_score(output, label)}")
print(predicted.shape)

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20,7))

prediction = predicted.detach().cpu().permute(1,2,0)
ground_truth = label.permute(1,2,0)
diff = (prediction != ground_truth)

fig.suptitle("Unet Segmentation Visualization Example")

axs[0].imshow(prediction)
axs[0].set_title("Prediction")
axs[1].imshow(ground_truth)
axs[1].set_title("Ground Truth")
axs[2].imshow(diff)
axs[2].set_title("Difference")

#plt.show()

#fig.savefig(parent_dir / "Image_Visualization" / "unet_seg_masks.png" )