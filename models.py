import math
import torch.nn as nn
import torch
from numpy import log2
from torch.nn.utils import spectral_norm


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_planes*2), GLU())
    return block

class Generator(nn.Module):
    def __init__(self, init_size, latent_dim, img_size, features, channels, layer):

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*features)

        super(Generator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.upscales = log2(img_size/self.init_size)
        


        self.network = UpBlock(latent_dim, self.init_size * self.features)


        self.normal_res_head = nn.Sequential(
            spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, 0), self.channels, 3, 1, 1))
        )

        self.high_res_block = UpBlock(int(self.init_size * self.features // math.pow(2,0)), self.init_size * self.features // pow(2,1))

        self.high_res_head = nn.Sequential(
            spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, 1), self.channels, 3, 1, 1))
        )

        for i in range(2,layer+1):
            self.add_layer(i)


    def forward(self, input, alpha):
        out = self.network(input)
        image_normal = self.normal_res_head(out)
        out_high = self.high_res_block(out)
        image_high = self.high_res_head(out_high)

        image_normal_high = nn.functional.interpolate(image_normal, size=image_high.size()[-2:],mode="bilinear", align_corners=False)
        image = (1-alpha) * image_normal_high + alpha * image_high

        return image

    def add_layer(self, layer):
        self.layer = layer
        self.network.extend(self.high_res_block)
        self.normal_res_head = self.high_res_head
        self.high_res_block = UpBlock(self.init_size * self.features // pow(2, self.layer-1), self.init_size * self.features // pow(2, self.layer))
        self.high_res_head = nn.Sequential(
            spectral_norm(nn.Conv2d(self.init_size * self.features // pow(2, self.layer), self.channels, 3, 1, 1))
        )

    def export_generative(self):
        final = self.network
        final.extend(self.high_res_block)
        final.extend(self.high_res_head)

        return final



def DownBlock(in_planes, out_planes):
    block = nn.Sequential(
        spectral_norm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False)),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

def DownBlockHead(in_planes, out_planes):
    block = nn.Sequential(
        spectral_norm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class Discriminator(nn.Module):
    def __init__(self, init_size, img_size, features, channels, layer):
        super(Discriminator, self).__init__()
        self.init_size = init_size
        self.features = features
        self.channels = channels
        self.layer = layer

        self.downscales = log2(img_size/self.init_size)



        self.network = DownBlock(int(self.init_size * self.features // math.pow(2,0)), 1)

        self.head_normal = DownBlockHead(self.channels, self.init_size * self.features // pow(2,0))


        self.head_high = DownBlockHead(self.channels, self.init_size * self.features // pow(2,1))

        self.body_high = DownBlock(self.init_size * self.features // pow(2,1), self.init_size * self.features // pow(2,0))
        for i in range(2,layer):
            self.add_layer(i)

    
    def forward(self, input_normal, input_high, alpha):
        pass_high = self.body_high(self.head_high(input_high)) # Downscale once
        pass_normal = self.head_normal(input_normal)
        input = (1 - alpha) * pass_normal + alpha * pass_high

        return self.network(input)

    def add_layer(self, layer):
        self.layer = layer
        self.network.insert(0, self.body_high)
        self.head_normal = self.head_high

        self.head_high = DownBlockHead(self.channels, self.init_size * self.features // pow(2, self.layer))
        self.body_high = DownBlock(self.init_size * self.features // pow(2,self.layer), self.init_size * self.features // pow(2,self.layer-1))
