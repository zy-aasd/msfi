import math

import torch
import torchvision
from torch import nn
from einops.layers.torch import Rearrange
from .PatchMerging import PatchMerging
class EncoderRes50(nn.Module):
    def __init__(self):
        super(EncoderRes50, self).__init__()
        self.down_layers = nn.ModuleList() # 双重下采样融合
        # CNN
        self.cnn_dims = [256, 512, 1024]
        self.patch_dims = [96, 192, 384]
        cnn_model = torchvision.models.resnet50(pretrained=True)
        self.resnet_layers = nn.ModuleList(cnn_model.children())[:7]
        # 1x1 Convs
        self.n1 = nn.Conv2d(in_channels=self.cnn_dims[0], out_channels=self.patch_dims[0], kernel_size=1)
        self.n2 = nn.Conv2d(in_channels=self.cnn_dims[1], out_channels=self.patch_dims[1], kernel_size=1)
        self.n3 = nn.Conv2d(in_channels=self.cnn_dims[2], out_channels=self.patch_dims[2], kernel_size=1)
        # Cnn Convs
        self.cnn2 = self.resnet_layers[5]
        self.cnn3 = self.resnet_layers[6]

        for i in range(len(self.patch_dims)): # 0~1  3 layers = 1 (cnn/patchEmbed fusion) + 2 (patchMerging/cnn fusion)
            self.down_layers.append(PatchMerging(dim=self.patch_dims[i]))

        self.layer_norm1 = nn.LayerNorm(self.patch_dims[0])
        self.layer_norm2 = nn.LayerNorm(self.patch_dims[1])
        self.layer_norm3 = nn.LayerNorm(self.patch_dims[2])

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 用于生成一个B,1,C的 CLS
        self.drop = nn.Dropout(0.1)
    def forward(self, x, x_patch):
        # x_patch = B,N,C
        # x = B,3,H,W
        B, N, C = x_patch.shape
        x_sml = [] # 多级特征图
        x_skip = [] # 跳跃连接
        # 最初的feature_size ，逐层//2
        feature_size = int(math.sqrt(N))  # 56
        for i in range(5):
            x = self.resnet_layers[i](x) # B,3,H,W->B 256 56 56

        # 第一层
        x_ = x  # 用于cnn层卷积的tensor # B 256 56 56
        x_fusion = self.n1(x_)  # B 128 56 56 -> # B 96 56 56

        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)

        x_patch = x_fusion + x_patch_fusion
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch) # B 56*56 96
        x_patch = self.layer_norm1(x_patch)
        x_sml.append(x_patch)

        feature_size = feature_size // 2  # 28
        # 第二层
        x_ = self.cnn2(x_)  # B 256 56 56 -> B 512 28 28
        x_fusion = self.n2(x_)  # B 512 28 28 -> # B 192 28 28

        x_patch = self.down_layers[0](x_patch)  # B 56*56 192  -> B 28*28 384
        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)

        x_patch = x_fusion + x_patch_fusion
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch) # B 28*28 192
        x_patch = self.layer_norm2(x_patch)

        pm_CLS = self.avg_pool(x_patch.transpose(1, 2))
        pm_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(pm_CLS)
        x_sml.append(torch.cat((pm_CLS_reshaped, x_patch), dim=1))

        feature_size = feature_size // 2  # 14

        # 第三层
        x_ = self.cnn3(x_)  # B 512 28 28 -> B 1024 14 14
        x_fusion = self.n3(x_)  # B 1024 14 14 -> # B 768 14 14

        x_patch = self.down_layers[1](x_patch)  # B 14*14 384 -> B 14*14 768

        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)
        x_patch = x_fusion + x_patch_fusion
        x_patch = self.drop(x_patch)
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch)  # B 14*14 384
        x_patch = self.layer_norm3(x_patch)
        ps_CLS = self.avg_pool(x_patch.transpose(1, 2))
        ps_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(ps_CLS)
        x_sml.append(torch.cat((ps_CLS_reshaped, x_patch), dim=1)) # B N+1 C
        return x_skip, x_sml


class EncoderRes34(nn.Module):
    def __init__(self):
        super(EncoderRes34, self).__init__()
        self.down_layers = nn.ModuleList() # 双重下采样融合
        # CNN
        self.cnn_dims = [128, 256, 512]
        self.patch_dims = [96, 192, 384]
        cnn_model = torchvision.models.resnet34(pretrained=True)
        self.resnet_layers = nn.ModuleList(cnn_model.children())[:10]
        # 1x1 Convs
        self.n1 = nn.Conv2d(in_channels=self.cnn_dims[0], out_channels=self.patch_dims[0], kernel_size=1)
        self.n2 = nn.Conv2d(in_channels=self.cnn_dims[1], out_channels=self.patch_dims[1], kernel_size=1)
        self.n3 = nn.Conv2d(in_channels=self.cnn_dims[2], out_channels=self.patch_dims[2], kernel_size=1)
        # Cnn Convs
        self.cnn2 = self.resnet_layers[6]
        self.cnn3 = self.resnet_layers[7]

        for i in range(len(self.patch_dims)): # 0~1  3 layers = 1 (cnn/patchEmbed fusion) + 2 (patchMerging/cnn fusion)
            self.down_layers.append(PatchMerging(dim=self.patch_dims[i]))

        self.layer_norm1 = nn.LayerNorm(self.patch_dims[0])
        self.layer_norm2 = nn.LayerNorm(self.patch_dims[1])
        self.layer_norm3 = nn.LayerNorm(self.patch_dims[2])

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 用于生成一个B,1,C的 CLS

        self.drop = nn.Dropout(0.1)
    def forward(self, x, x_patch):
        # x_patch = B,N,C
        # x = B,3,H,W
        B, N, C = x_patch.shape
        x_sml = [] # 多级特征图
        x_skip = [] # 跳跃连接
        # 最初的feature_size ，逐层//2
        feature_size = int(math.sqrt(N))  # 56
        for i in range(6):
            x = self.resnet_layers[i](x) # B,3,H,W->B 128 28 28

        # 第一层
        x_ = x  # 用于cnn层卷积的tensor # B 128 28 28
        x_fusion = self.n1(x_)  # B 128 28 28 -> # B 96 28 28
        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)

        x_patch = x_fusion + x_patch_fusion
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch) # B 28*28 96
        x_patch = self.layer_norm1(x_patch)
        x_sml.append(x_patch)
        feature_size = feature_size // 2  # 28
        # 第二层
        x_ = self.cnn2(x_)  # B 128 28 28 -> B 256 14 14
        x_fusion = self.n2(x_)  # B 256 14 14 -> # B 192 14 14

        x_patch = self.down_layers[0](x_patch)  # B 28*28 192  -> B 14*14 192
        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)

        x_patch = x_fusion + x_patch_fusion
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch) # B 14*14 192
        x_patch = self.layer_norm2(x_patch)

        pm_CLS = self.avg_pool(x_patch.transpose(1, 2))
        pm_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(pm_CLS)
        x_sml.append(torch.cat((pm_CLS_reshaped, x_patch), dim=1))


        feature_size = feature_size // 2  # 14

        # 第三层
        x_ = self.cnn3(x_)  # B 256 14 14 -> B 512 7 7
        x_fusion = self.n3(x_)  # B 512 7 7 -> # B 384 7 7

        x_patch = self.down_layers[1](x_patch)  # B 14*14 192 -> B 7*7 384

        x_patch_fusion = Rearrange('b (h w) c -> b c h w', h=feature_size, w=feature_size)(x_patch)
        x_patch = x_fusion + x_patch_fusion
        x_patch = self.drop(x_patch)
        x_skip.append(x_patch)
        x_patch = Rearrange('b c h w -> b (h w) c')(x_patch)  # B 14*14 384
        x_patch = self.layer_norm3(x_patch)
        ps_CLS = self.avg_pool(x_patch.transpose(1, 2))
        ps_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(ps_CLS)
        x_sml.append(torch.cat((ps_CLS_reshaped, x_patch), dim=1))

        return x_skip, x_sml
