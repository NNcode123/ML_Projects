import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from DNN_MODEL.model import BASIC_MODEL


def main():
    model = BASIC_MODEL()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(torch.load(Path("checkpoint") / "model.pth"))

    model.to(torch.device(device))

    BATCH  = 128

    transform = transforms.ToTensor()

    MNIST_Test = datasets.MNIST(root = "data", train = False, transform = transform )

    test_data_loader = DataLoader(dataset = MNIST_Test, batch_size = BATCH//2, shuffle = True, num_workers = 2)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():

        loss_num, accuracy, total_batches = 0, 0, 0

        for (image,label) in test_data_loader:

            image, label = image.to(device), label.to(device)

            output = model(image.view(image.shape[0], -1))

            loss = criterion(output,label)

            loss_num += loss.item()

            #print(output.shape)

            accuracy_cur = (output.argmax(dim = 1) == label).sum().item()

            total_batches += label.numel()

            accuracy += accuracy_cur

        print(f"Accuracy: {accuracy/total_batches}, Loss: {loss_num/len(test_data_loader)}")



if __name__ == "__main__":
    main()