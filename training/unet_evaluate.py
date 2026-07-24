from .unet_train import test_data
from torch.utils.data import DataLoader
from utils.parent_dir import parent_dir
from utils.device import device
from utils.metrics import calc_accuracy, miou_score, mdice_score
from Models.model import UNET
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 



def main():

    test_data_loader  = DataLoader(dataset = test_data, batch_size  =4, shuffle = True, num_workers = 4)

    model_state_dict = torch.load(parent_dir/ "checkpoint" / "unet" / "epoch_74.pth")

    model = UNET().to(device)

    model.load_state_dict(model_state_dict["model"])

    criterion = nn.CrossEntropyLoss()

    model.eval()

    tot_acc, tot_IOU, tot_DICE, tot_loss = (0,0,0,0)

    with torch.no_grad():

        

        for (index, (image, label)) in enumerate(test_data_loader):

            image,label = image.to(device), label.to(device)

            label = (label.squeeze(1)).long()

            output = model(image)

            if (index <= 5):
                print(f"output_shape: {output.shape}, label_shape: {label.shape}")

                print(f"label_unique: {torch.unique(label.detach())},  tens_argmax_uniqe: {torch.unique(output.detach().argmax(axis = 1))}")

            loss = criterion(output, label - 1)

            tot_loss += loss.item()

            tot_acc += calc_accuracy(output, label-1)

            tot_IOU += miou_score(output, label)

            tot_DICE += mdice_score(output, label)
    


    avg_loss = tot_loss / len(test_data_loader)
    avg_acc = tot_acc / len(test_data_loader)
    avg_IOU = tot_IOU / len(test_data_loader)
    avg_DICE = tot_DICE / len(test_data_loader)

    print(f"Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | mIOU: {avg_IOU:.4f} | DICE: {avg_DICE:.4f}")

    new_dir = parent_dir / "Results"

    result_file = new_dir/ "UNET_CrossEntropyLoss_Seg_Loss_Results.png"

    new_dir.mkdir(exist_ok = True)

    result_file.touch(exist_ok = True)

    new_dir.mkdir(exist_ok=True)

    metrics = [
        "Loss",
        "Accuracy",
        "mIoU",
        "Dice"
    ]

    values = [
        avg_loss,
        avg_acc,
        avg_IOU,
        avg_DICE
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(metrics, values)

    plt.ylabel("Score")
    plt.title("Unet Segmentation Performance Metrics")

    # Add values above bars
    for i, value in enumerate(values):
        plt.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0, 1.1)
    plt.legend(["Metric"])

    plt.tight_layout()

    plt.savefig(result_file, dpi=300)
    plt.close()



if __name__ == "__main__":
    main()