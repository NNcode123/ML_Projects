
import torch.optim
import os 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import sys
import numpy
import matplotlib.pyplot as plt


"""



X = torch.randn(8).to('cuda')
Y = torch.randn(8).to('cuda')
W = torch.randn(8,8, device = 'cuda', requires_grad = True)
B = torch.randn(8, device = 'cuda', requires_grad = True)

print(W.is_leaf, B.is_leaf)

"""



def model(W,input,bias) -> torch.Tensor:
    return W@input+bias




"""
for epoch in range(EPOCHS):

    output = model(W,X,B)

    loss_tensor = ((Y-output)**2).mean()

    loss_tensor.backward()

    with torch.no_grad():

        W  -= LR * W.grad

        B -= LR* B.grad

    
    W.grad.zero_()
    B.grad.zero_()
    
    loss_value = loss_tensor.item()

    print(f"Epoch: {epoch}, loss: {loss_value:.4f}")
    """







def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    BATCH = 64
    LR = 0.005
    EPOCHS = 150

    class BASIC_MODEL(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers = nn.Sequential(nn.Linear(28*28, 512),  nn.ReLU(inplace = True), nn.Dropout(0.2),  
                                        nn.Linear(512, 30), nn.ReLU(inplace = True), nn.Dropout(0.2),
                                        nn.Linear(30,10))

        def forward(self, x) -> torch.Tensor:
            return self.layers(x)

    model = BASIC_MODEL().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    transform = transforms.ToTensor()


    MNIST_Train = datasets.MNIST(root = "data", train= True,  transform = transform)

    MNIST_Test = datasets.MNIST(root = "data", train = False, transform = transform )


    train_data_loader = DataLoader(dataset = MNIST_Train, batch_size= BATCH, shuffle = True, num_workers = 0)

    #test_data_loader = DataLoader(dataset = MNIST_Test, batch_size = 1000, num_workers = 2)

    print(f"num_parameters: {sum(p.numel() for p in model.parameters())}")


    for epoch in range(EPOCHS):

        model.train()
        
        train_loss, train_acc, index = 0, 0, 0


        for (image,label) in train_data_loader:
            
           


            image, label = image.to(device), label.to(device)

            output = model(image.view(image.shape[0],-1))

        
           # print(f"Predicted_Class_shape: {output.shape}, Actual_Class_shape: {label.shape} ")
            
            
            
            loss= criterion(output, label)

            optimizer.zero_grad()

            loss.backward()

            train_loss += loss.item()

            optimizer.step()

            index += 1

        
        
        print(f"Epoch: {epoch}, Loss: {train_loss/len(train_data_loader):.4f}")


if __name__ == "__main__":
    main()


    

