import torch.nn as nn
import torch

LR = 0.001
BATCH_SIZE = 20

ROOT = '/home/pari/Projects/Research/Tensor_CS/data_drive/'
# ROOT = '/nfs/stak/users/shressag/sagar/deep_completion/data/'

image_size = 51
nc = 2
ndf = 32
ngpu = 1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))

class EncoderSelu(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.SELU(),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.SELU(),

            Flatten()
        )

    def forward(self, input):
        # for layer in self.main:
        #     input = layer(input)
        #     print(input.shape)
        # return input
        return self.main(input)

class DecoderSelu(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            
            UnFlatten((ndf*16, 1, 1)),
            # state size. (ndf*16) x 1 x 1
            nn.ConvTranspose2d(ndf*16, ndf*8, 3, 1, 0),
            nn.BatchNorm2d(ndf*8),
            nn.SELU(),
            
            # state size. (ndf*8) x 3 x 3
            nn.ConvTranspose2d(ndf*8, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6   
            nn.ConvTranspose2d(ndf*4, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.SELU(),

            # state size (ndf*2) x 12 x 12
            nn.ConvTranspose2d(ndf*2, ndf*1, 4, 2, 0),
            nn.BatchNorm2d(ndf*1),
            nn.SELU(),

            # state size (ndf) x 26, 26
            nn.ConvTranspose2d(ndf*1,2, 4, 2, 0),
            nn.BatchNorm2d(2),
            nn.SELU(),

            # state size 2 x 54 x 54
            nn.Conv2d(2,1, 4, 1, 0),
            nn.Sigmoid()
            # output 1 x 51 x 51            
        )

    def forward(self, input):
        # for layer in self.main:
        #     input = layer(input)
        #     print(input.shape)
        # return input
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, ndf * 16, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            Flatten()
        )

    def forward(self, input):
        # for layer in self.main:
        #     input = layer(input)
        #     print(input.shape)
        # return input
        return self.main(input)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 16
        self.main = nn.Sequential(
            
            UnFlatten((ndf*16, 1, 1)),
            # state size. (ndf*16) x 1 x 1
            nn.ConvTranspose2d(ndf*16, ndf*8, 3, 1, 0),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            
            # state size. (ndf*8) x 3 x 3
            nn.ConvTranspose2d(ndf*8, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            
            # state size. (ndf*4) x 6 x 6   
            nn.ConvTranspose2d(ndf*4, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),

            # state size (ndf*2) x 12 x 12
            nn.ConvTranspose2d(ndf*2, ndf*1, 4, 2, 0),
            nn.BatchNorm2d(ndf*1),
            nn.ReLU(True),

            # state size (ndf) x 26, 26
            nn.ConvTranspose2d(ndf*1,2, 4, 2, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(True),

            # state size 2 x 54 x 54
            nn.Conv2d(2,1, 4, 1, 0),
            nn.Sigmoid()
            # output 1 x 51 x 51            
        )

    def forward(self, input):
        # for layer in self.main:
        #     input = layer(input)
        #     print(input.shape)
        # return input
        return self.main(input)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        z = self.encoder(input)
        return self.decoder(z)

class AutoencoderSelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderSelu()
        self.decoder = DecoderSelu()

    def forward(self, input):
        z = self.encoder(input)
        return self.decoder(z)

