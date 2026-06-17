
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
     


class CNN_CAT_DOG(nn.Module):
     def __init__(self):
          
          """
          input shape: (Batch, Channel, Height, Width) -> (Batch, 3, 128, 128)
          
          """

          super().__init__()
          
          self.features = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1 ),
                                        nn.ReLU(inplace = True),
                                        nn.BatchNorm2d(8),
                                        nn.MaxPool2d(2),
                                        #Output shape here will be (Batch, 16, 64, 64)
                                    
                                        nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1 ),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(inplace = True),
                                        
                                        nn.MaxPool2d(2),

                                        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1 ),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace = True),
                                        nn.MaxPool2d(2),
                                        #Output shape here wil be (Batch, 16,32, 32  )

                                       )

                                        
                                        #Output shape here will be (Batch, 32, 16, 16))

          self.linear = nn.Sequential(nn.Flatten(), nn.Linear(in_features = 32*16*16, out_features = 64))
          
          self.classifier = nn.Sequential(nn.Linear(in_features = 64, out_features = 2 ) )
                                       
                                        






          


     def forward(self, x) -> torch.Tensor:
          return self.classifier(self.linear(self.features(x)))
     

     def feature_map(self, x) ->torch.Tensor:
          return self.features(x)
     


class Encoder(nn.Module):
     def __init__(self, inCh: int, outCh: int):
          super().__init__()
          self.features =      nn.Sequential(nn.Conv2d(in_channels =inCh , out_channels = outCh, kernel_size = 3, padding = 1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = outCh, out_channels = outCh, kernel_size = 3, padding = 1),nn.ReLU())
                                   
                                   
          self.pool  =   nn.MaxPool2d(2 )
     
     def forward(self,x):
          x = self.features(x)
          y = self.pool(x)
          return (x,y)
     

class Decoder(nn.Module):
     def __init__(self, inCh: int, outCh: int):
          super().__init__()
          self.layers = nn.Sequential(nn.Conv2d(in_channels =inCh , out_channels = outCh, kernel_size = 3, padding = 1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = outCh, out_channels = outCh, kernel_size = 3, padding =1),nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels = outCh, out_channels = outCh//2, kernel_size = 2, stride = 2 ))
     
     def forward(self,x):
          return self.layers(x)
     



class UNET(nn.Module):
     def __init__(self):

          super().__init__()
     
          self.enc1 = Encoder(inCh=3,   outCh=16)
          self.enc2 = Encoder(inCh=16,  outCh=32)
          self.enc3 = Encoder(inCh=32,  outCh=64)
          self.enc4 = Encoder(inCh=64,  outCh=128)

          self.layer = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
          )

          self.dec1 = Decoder(inCh=256, outCh=128)
          self.dec2 = Decoder(inCh=128, outCh=64)
          self.dec3 = Decoder(inCh=64,  outCh=32)

          self.seg_map_seq = nn.Sequential(
          nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
          )
          

     

     def forward(self, x: torch.Tensor) -> torch.Tensor:
          skip4 , f1 = self.enc1(x)
          skip3, f2 = self.enc2(f1)
          skip2, f3 = self.enc3(f2)
          skip1, f4 = self.enc4(f3)
          f5 = self.layer(f4)
          up1 = torch.cat((skip1, f5), dim = 1)
          up2 = torch.cat((skip2, self.dec1(up1)), dim = 1)
          up3 = torch.cat((skip3, self.dec2(up2)), dim = 1)
          up4 = torch.cat((skip4, self.dec3(up3)), dim = 1)

          return self.seg_map_seq(up4)
          
         



"""
                                        
                                        """