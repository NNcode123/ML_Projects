import matplotlib.pyplot as plt
from pathlib import Path
from training.unet_train import train_data
import torch.utils.data as data
import torch
from Models.model import UNET
from Models.deeplabV3_ResNet50 import DeepLabV3
from utils.device import device
from utils.parent_dir import parent_dir


train_data_loader = data.DataLoader(dataset=train_data, batch_size=64, num_workers=0, shuffle=True)

image_batch, labels = next(iter(train_data_loader))
images = image_batch[:3]
labels = labels[:3]



unet_model = UNET().to(device)
deep_model = DeepLabV3().to(device)

unet_checkpoint = torch.load(Path("checkpoint") / "unet" / "epoch_74.pth", map_location=device)
deep_checkpoint = torch.load(Path("checkpoint") / "DeepLabV3" / "epoch_34.pth", map_location=device)

unet_model.load_state_dict(unet_checkpoint["model"])
deep_model.load_state_dict(deep_checkpoint["model"])

unet_model.eval()
deep_model.eval()

with torch.no_grad():
    image_inputs = images.to(device)
    unet_output = unet_model(image_inputs)
    deep_output = deep_model(image_inputs)
    combined_output = unet_output + deep_output

unet_pred = unet_output.argmax(dim=1).cpu()
deep_pred = deep_output.argmax(dim=1).cpu()
combined_pred = combined_output.argmax(dim=1).cpu()


fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(24, 12))
fig.suptitle("Segmentation Visualization Comparison")

for row_idx in range(3):
    image = images[row_idx].permute(1, 2, 0).cpu().numpy()
    ground_truth = labels[row_idx].squeeze().numpy()
    unet_mask = unet_pred[row_idx].numpy()
    deep_mask = deep_pred[row_idx].numpy()
    combined_mask = combined_pred[row_idx].numpy()
    diff = (combined_mask != ground_truth)

    axs[row_idx, 0].imshow(image)
    axs[row_idx, 0].set_title("Input Image")
    axs[row_idx, 0].axis("off")

    axs[row_idx, 1].imshow(ground_truth)
    axs[row_idx, 1].set_title("Ground Truth")
    axs[row_idx, 1].axis("off")

    axs[row_idx, 2].imshow(unet_mask)
    axs[row_idx, 2].set_title("UNet Prediction")
    axs[row_idx, 2].axis("off")

    axs[row_idx, 3].imshow(deep_mask)
    axs[row_idx, 3].set_title("DeepLabV3 Prediction")
    axs[row_idx, 3].axis("off")

    axs[row_idx, 4].imshow(combined_mask)
    axs[row_idx, 4].set_title("Combined Prediction")
    axs[row_idx, 4].axis("off")

    axs[row_idx, 5].imshow(diff)
    axs[row_idx, 5].set_title("Difference")
    axs[row_idx, 5].axis("off")

    axs[row_idx, 0].set_ylabel(f"Sample {row_idx + 1}")

plt.tight_layout()
plt.show()

image_dir = parent_dir/"Image_Visualization"/"unet_DeeplabV3_seg_comparison_mask.png"
image_dir.touch(exist_ok = True)


plt.savefig(image_dir)