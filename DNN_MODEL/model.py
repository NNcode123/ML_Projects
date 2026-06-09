
import torch
from torch import nn

class BASIC_MODEL(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers = nn.Sequential(nn.Linear(28*28, 512),  nn.ReLU(inplace = True), nn.Dropout(0.2),  
                                        nn.Linear(512, 30), nn.ReLU(inplace = True), nn.Dropout(0.2),
                                        nn.Linear(30,10))

        def forward(self, x) -> torch.Tensor:
            return self.layers(x)
        



class BASIC_CNN(nn.Module):
     
    def __init__(self):
        """"
        Input Shape: (Batch_size, channels (1 means GrayScale, 3 means RGB), Height, Width)

        Output Shape: (Batch_Size, Num Classes)

        """
        super().__init__()

        self.features  = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1 ),
                                       nn.ReLU(inplace = True), 
                                       nn.MaxPool2d(2),
                                       nn.Conv2d(in_channels = 16,  out_channels = 32, kernel_size = 3, padding = 1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(2))

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features = 32 * 7 *7, out_features = 10 ) )


        
    
    
    def forward(self, x) -> torch.Tensor:
        return self.classifier(self.features(x))
     