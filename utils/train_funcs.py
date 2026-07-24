
import torch.nn as nn
from torch import optim
import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path 
from utils.metrics import calc_accuracy 
from utils.metrics import miou_score, mdice_score 
from utils.parent_dir import parent_dir
from utils.device import device

def train_results(BATCH: int, LR: float, EPOCHS:int , model: nn.Module, dataset: torchvision.datasets, batch_prep_func, save_path: Path):

        # Initialize the model, loss function, optimizer, and data transform
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        # Load the MNIST training dataset from the local data folder

        train_loader = DataLoader(dataset = dataset, batch_size= BATCH, shuffle = True, num_workers = 0)
        
        # Training loop over epochs and batches
        for epoch in range(EPOCHS):

            model.train()
            
            train_loss, train_acc = 0, 0


            for (image,label) in train_loader:
                
                image, label = image.to(device), label.to(device)

                image = batch_prep_func(image) if batch_prep_func else image

                output = model(image)

                loss= criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                train_acc += calc_accuracy(output, label)

                train_loss += loss.item()

                

            
            
            print(f"Epoch: {epoch}, train_accuracy: {train_acc/len(train_loader):.4f}, train_Loss: {train_loss/len(train_loader)}")

    

            Path.touch(save_path, exist_ok = True)

            torch.save({"model": model.state_dict(),
                        "optim": optimizer.state_dict()}, save_path)

            
def train_seg_model(
    model: nn.Module,
    train_dataset: torchvision.datasets,
    optimizer, 
    criterion,
    batch_size: int,
    epochs: int,
    checkpoint_dir: Path,
    device: torch.device,
    start_epoch=-1,
):

    model = model.to(device)


    if start_epoch != -1:

        checkpoint_file = (
            checkpoint_dir
            / f"epoch_{start_epoch - 1}.pth"
        )

        try:

            checkpoint = torch.load(checkpoint_file)

            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optim"])

        except FileNotFoundError:

            raise RuntimeError(
                f"Checkpoint not found: {checkpoint_file}"
            )
        

            

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    model.train()

    

    epoch_range = (
        range(start_epoch, start_epoch + epochs)
        if start_epoch != -1
        else range(epochs)
    )

    

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epoch_range:

        total_loss = 0
        total_acc = 0
        total_mIOU = 0
        total_mDICE = 0

        for image, label in train_loader:

            image = image.to(device)
            label = label.to(device).squeeze(1).long()

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(
                output,
                label-1,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total_acc += calc_accuracy(
                output,
                label - 1,
            )

            total_mIOU += miou_score(output, label)
            total_mDICE += mdice_score(output, label)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        avg_dice = total_mDICE/len(train_loader)
        avg_mIOU = total_mIOU/len(train_loader)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "accuracy": avg_acc,
                "loss": avg_loss,
                "IOU": avg_mIOU,
                "DICE": avg_dice
            },
            checkpoint_dir
            / f"epoch_{epoch}.pth",
        )

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | IOU: {avg_mIOU:.4f} | DICE: {avg_dice:.4f}")
        

        
        