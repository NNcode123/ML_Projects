import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

class rNet_block(nn.Module):
    def __init__(self, in_c, mid_c, out_c, strde = 1, proj = False, dil = 1):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = mid_c, kernel_size = 1 , stride = strde, dilation = dil),
                                   nn.BatchNorm2d(mid_c),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels = mid_c, out_channels = mid_c, kernel_size = 3, padding = 1), 
                                   nn.BatchNorm2d(mid_c),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels = mid_c, out_channels = out_c, kernel_size = 1),
                                   nn.BatchNorm2d(out_c))
        
        self.proj = nn.Sequential(nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 1, stride = strde, dilation = dil, bias = False ), 
                                  nn.BatchNorm2d(out_c)) if proj else nn.Identity()

        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.block(x), self.proj(x)
        print(x.shape, y.shape)
        return self.activation(x+y)
    

class ResNet50_backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7,  stride = 2, padding = 3))

        layers = [rNet_block(in_c = 64, mid_c = 64, out_c = 256, strde = 2, proj = True) if _ == 0 else rNet_block(in_c = 256, mid_c = 64, out_c = 256 )for _ in range(3)]

        self.block1 = nn.Sequential(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1 ), *layers)

        layers = [rNet_block(in_c = 256, mid_c = 128, out_c = 512, strde = 2, proj = True) if _ == 0 else rNet_block(in_c = 512, mid_c = 128, out_c = 512)  for _ in range(4)]

        self.block2 = nn.Sequential(*layers)

        layers = [rNet_block(in_c = 512, mid_c = 256, out_c = 1024, strde = 2, proj = True) if _ == 0 else rNet_block(in_c = 1024, mid_c = 256, out_c = 1024)  for _ in range(6)]

        self.block3 = nn.Sequential(*layers)

        layers = [rNet_block(in_c = 1024, mid_c = 512, out_c = 2048,  proj = True, dil = 2) if _ == 0 else rNet_block(in_c = 2048, mid_c = 512, out_c = 2048) for _ in range(3)]

        self.block4 = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return x
    

class ASSP_Module(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 2048, out_channels = 256, kernel_size = 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU())


        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 2048, out_channels = 256, kernel_size = 3, dilation  = 6, padding = 6 ),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 2048, out_channels  = 256, kernel_size = 3, dilation = 12, padding  = 12  ), 
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())

        self.conv4 =    nn.Sequential( nn.Conv2d(in_channels = 2048, out_channels = 256, kernel_size = 3, dilation =  18, padding = 18  ),
                                nn.BatchNorm2d(256),
                                nn.ReLU())

        self.poolSeg= nn.Sequential(nn.AvgPool2d(1), nn.Conv2d(in_channels = 2048, out_channels = 256, kernel_size = 1, bias = False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        

        
        self.proj = nn.Sequential( nn.Conv2d(in_channels = 256*5, out_channels = 256, kernel_size = 1, bias = False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        
        self.classifier = nn.Conv2d(in_channels = 256, out_channels = 3, kernel_size = 3, padding = 1,  bias = False)
        

    def forward(self, x: torch.Tensor):

        x_cat = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x), self.poolSeg(x)], dim = 1)
        x_cat = self.proj(x_cat)
        return self.classifier(F.interpolate(x_cat, scale_factor = 32, mode= 'bilinear'))



class DeepLabV3(nn.Module):
    def __init__(self):

        super().__init__()

        self.RNET_Backbone = ResNet50_backbone()

        self.ASPP = ASSP_Module()

    
    def forward(self, x):
        x = self.RNET_Backbone(x)
        x = self.ASPP(x)

         
    

if __name__ == "__main__":

    input = torch.randn(64,3, 256, 256).to('cuda')

    #output = ResNet50_backbone().to('cuda') (input.to('cuda'))

    summary(DeepLabV3().to(torch.device('cuda')), input_data = input)




        



                                    