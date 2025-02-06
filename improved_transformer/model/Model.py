import copy
import math

import torch
from torch import nn
from .PatchEmbed import PatchEmbed
from einops.layers.torch import Rearrange
from .Former import Former
from .Decoder import SegmentationHead, Decoder, DecoderExpand, DecoderBilinear, DecoderTranspose2d, DecoderExpandConv
from .Encoder import *
class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, num_classes=3, pretrain='res34', norm_layer=nn.LayerNorm, upsample=None):
        super(Model, self).__init__()
        #
        if pretrain == 'res50':
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,  # 224 / 4 = 56, 256/4=64
                                          embed_dim=embed_dim, norm_layer=nn.LayerNorm)
            resolution = img_size // patch_size
            self.encoder = EncoderRes50()
            # 3135 = 56*56-1 784=28*28  195=14*14
            num_patches = [resolution*resolution-1, resolution*resolution//4, resolution*resolution//16]
            upsample = False
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size * 2, in_chans=in_chans, # 224 / 8 = 28
                                          embed_dim=embed_dim, norm_layer=nn.LayerNorm)
            self.encoder = EncoderRes34()
            resolution = img_size // patch_size
            num_patches = [resolution * resolution - 1, resolution * resolution // 4, resolution * resolution // 16]
            upsample = True


        self.former = Former(qkv_bias=False, num_patches=num_patches, drop=0., norm_layer=nn.LayerNorm)

        self.norm = norm_layer(embed_dim * 4)

        self.decoder = Decoder(embed_dim * 4, out_chans=[embed_dim * 2, embed_dim], upsample=upsample)

        self.segmentationHead = SegmentationHead(96, num_classes)

    def forward(self, x):
        # B 3 H W
        x_patch = self.patch_embed(x)  # B N C
        x_skip, x_sml = self.encoder(x, x_patch)  # B,N,C -> B, H/16*W/16 ,4d

        x = self.former(x_sml)
        x = x[:, 1:, :]
        x = self.norm(x)
        B, N, C = x.shape
        hw = int(math.sqrt(N))
        x = Rearrange("b (h w) c -> b c h w", h=hw, w=hw)(x)

        x = self.decoder(x, x_skip)
        x = self.segmentationHead(x)
        return x


if __name__ == '__main__':
    tensor = torch.randn(1,3,256,256).cuda()
    model = Model(img_size=256).cuda()
    output = model(tensor).cuda()

