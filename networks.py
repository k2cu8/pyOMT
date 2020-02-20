import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import pdb

class autoencoder(nn.Module):
    def __init__(self, dim_z=100, dim_c=3, dim_f=16):
        super(autoencoder, self).__init__()
        self.dim_c = dim_c
        self.dim_z = dim_z


        self.block1 = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            nn.Conv2d(3, dim_f, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block2 = nn.Sequential(
            # [-1, 256, 8, 8]
            nn.Conv2d(dim_f, dim_f * 2, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block3 = nn.Sequential(
            # [-1, 512, 4, 4]
            nn.Conv2d(dim_f * 2, dim_f * 4, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(dim_f * 4, dim_f * 8, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.block5 = nn.Sequential(
            # [-1, 1 + cc_dim + dc_dim, 1, 1]
            nn.Conv2d(dim_f * 8, dim_z, 4, 1, 0)
        )
   

        self.block6 = nn.ConvTranspose2d(dim_z, dim_f * 8, 4, 1, 0)

        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 8, dim_f * 4, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 4),
            nn.ReLU(),
        )

        self.block8 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 4, dim_f * 2, 4, 2, 1),
            nn.BatchNorm2d(dim_f * 2),
            nn.ReLU(),
        )

        self.block9 = nn.Sequential(
            nn.ConvTranspose2d(dim_f * 2, dim_f, 4, 2, 1),
            nn.BatchNorm2d(dim_f),
            nn.ReLU(),
        )

        self.block10 = nn.Sequential(
            nn.ConvTranspose2d(dim_f, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encoder(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    def decoder(self, z):
        x6 = self.block6(z)
        x7 = self.block7(x6)
        x8 = self.block8(x7)
        x9 = self.block9(x8)
        x10 = self.block10(x9)
        return x10
    
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7)
        x9 = self.block9(x8)
        x10 = self.block10(x9)
        return x10, x5

