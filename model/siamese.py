from torch import nn
from model.blocks import Conv2dReLU


class Siamese(nn.Module):
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.conv_1 = Conv2dReLU(1, 512, kernel_size=(80, 3))
        self.conv_2 = Conv2dReLU(512, 512, kernel_size=(1, 9), stride=(1, 2))
        self.conv_2 = Conv2dReLU(512, 512, kernel_size=(1, 7), stride=(1, 2))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(333312, latent_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x