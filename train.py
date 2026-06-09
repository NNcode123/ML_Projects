
import torch.optim
import random
import os 
import argparse as arg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from DNN_MODEL.model import BASIC_MODEL
from DNN_MODEL.model import BASIC_CNN

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



    def train_results(BATCH, LR, EPOCHS):

        model = BASIC_CNN().to(torch.device(device))

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr = LR)

        transform = transforms.ToTensor()


        MNIST_Train = datasets.MNIST(root = "data", train= True,  transform = transform)

        


        train_data_loader = DataLoader(dataset = MNIST_Train, batch_size= BATCH, shuffle = True, num_workers = 0)

        


        

        print(f"num_parameters: {sum(p.numel() for p in model.parameters())}")



        

        for epoch in range(EPOCHS):

            model.train()
            
            train_loss, train_acc, index = 0, 0, 0


            for (image,label) in train_data_loader:
                
            


                image, label = image.to(device), label.to(device)

                #output = model(image.view(image.shape[0],-1))

                output = model(image)

            
            # print(f"Predicted_Class_shape: {output.shape}, Actual_Class_shape: {label.shape} ")
                
                
                
                loss= criterion(output, label)

                optimizer.zero_grad()

                loss.backward()

                train_loss += loss.item()

                optimizer.step()

                index += 1

            
            
            print(f"Epoch: {epoch}, Loss: {train_loss/len(train_data_loader):.4f}")

        images, labels = next(iter(train_data_loader))

        rand_ind_start = random.randint(0, 31)


        image = images[rand_ind_start,...].squeeze()

        plt.imshow(image)

        plt.show()

        model_path = Path("checkpoint") /"model.pth"

        optimizer_path = Path("checkpoint") /"optimizer.pth"

        Path.touch(model_path)

        Path.touch(optimizer_path)

        torch.save(model.state_dict(), model_path)

        torch.save(optimizer.state_dict(), optimizer_path)  


    

    arg_collector = arg.ArgumentParser()

    arg_collector.add_argument(  "--lr", type = float)
    arg_collector.add_argument("--batch_size", type = int)
    arg_collector.add_argument("--epochs", type = int)


    args = arg_collector.parse_args()

    train_results(args.batch_size, args.lr, args.epochs)








    








    

    

    
    




if __name__ == "__main__":
    main()


    

