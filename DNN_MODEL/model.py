
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