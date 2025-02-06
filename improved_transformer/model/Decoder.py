import torch
from torch import nn
from .PatchExpand import *
class Decoder(nn.Module):
    def __init__(self, in_chans=384, out_chans=[192, 96], upsample=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans[0], kernel_size=2, stride=2), # 384 -> 192
            nn.BatchNorm2d(out_chans[0]),
            nn.GELU(),
            nn.Conv2d(out_chans[0], out_chans[0], kernel_size=3, stride=1, padding=1),  # 192 -> 192
            nn.BatchNorm2d(out_chans[0]),
            nn.GELU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(out_chans[0], out_chans[1], kernel_size=2, stride=2), # 192 -> 96 = Upsample 2
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
            nn.Conv2d(out_chans[1], out_chans[1],  kernel_size=3, stride=1, padding=1),  # 96 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
        )
        self.conv_up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(out_chans[1], out_chans[1], kernel_size=4, stride=4),  # 96 -> 96 = Upsample 4
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),

            nn.Conv2d(out_chans[1], out_chans[1], kernel_size=1, stride=1), # 96 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU()
        )
        self.conv1 = nn.Conv2d(out_chans[0], out_chans[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_chans[1], out_chans[1], kernel_size=1, stride=1)

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

    def forward(self, x, x_skip):
        # x 384
        # x_skip [384, 192, 96]
        x = x + x_skip[2]
        x1 = self.conv_layer1(x)
        t = x1
        x1 = self.conv1(t)
        x = x1 + t

        x = x + x_skip[1]
        x2 = self.conv_layer2(x)
        t = x2
        x2 = self.conv2(t)
        x = x2 + t

        x = x + x_skip[0]
        x3 = self.conv_up(x)
        t = x3
        x3 = self.conv2(t)
        x = x3 + t

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class DecoderExpand(nn.Module):
    def __init__(self, in_chans=384, out_chans=[192, 96], upsample=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_layer1 = nn.Sequential(
            PatchExpand(input_resolution=[14,14], dim=in_chans), # 384 -> 192
            nn.GELU(),
        )
        self.conv_layer2 = nn.Sequential(
            PatchExpand(input_resolution=[28,28], dim=out_chans[0]), # 192 -> 96 = Upsample 2
            nn.GELU(),
        )
        self.conv_up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            FinalPatchExpand_X4(input_resolution=[56, 56], dim=out_chans[1]),  # 96 -> 96 = Upsample 4
            nn.GELU(),
        )

    def forward(self, x, x_skip):
        # x 384
        # x_skip [384, 192, 96]
        x = x + x_skip[2]
        x = self.conv_layer1(x)

        x = x + x_skip[1]
        x = self.conv_layer2(x)

        x = x + x_skip[0]
        x = self.conv_up(x)
        return x

class DecoderExpandConv(nn.Module):
    def __init__(self, in_chans=384, out_chans=[192, 96], upsample=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_layer1 = nn.Sequential(
            PatchExpand(input_resolution=[14,14], dim=in_chans), # 384 -> 192
            nn.GELU(),
            nn.Conv2d(out_chans[0], out_chans[0], kernel_size=3, stride=1, padding=1),  # 192 -> 192
            nn.BatchNorm2d(out_chans[0]),
            nn.GELU(),
        )
        self.conv_layer2 = nn.Sequential(
            PatchExpand(input_resolution=[28,28], dim=out_chans[0]), # 192 -> 96 = Upsample 2
            nn.GELU(),
            nn.Conv2d(out_chans[1], out_chans[1], kernel_size=3, stride=1, padding=1),  # 96 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
        )
        self.conv_up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            FinalPatchExpand_X4(input_resolution=[56, 56], dim=out_chans[1]),  # 96 -> 96 = Upsample 4
            nn.GELU(),
            nn.Conv2d(out_chans[1], out_chans[1], kernel_size=1, stride=1),  # 96 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
        )
        self.conv1 = nn.Conv2d(out_chans[0], out_chans[0], kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_chans[1], out_chans[1], kernel_size=1, stride=1)

    def forward(self, x, x_skip):
        # x 384
        # x_skip [384, 192, 96]
        x = x + x_skip[2]
        x1 = self.conv_layer1(x)
        t = x1
        x1 = self.conv1(t)
        x = x1 + t

        x = x + x_skip[1]
        x2 = self.conv_layer2(x)
        t = x2
        x2 = self.conv2(t)
        x = x2 + t

        x = x + x_skip[0]
        x3 = self.conv_up(x)
        t = x3
        x3 = self.conv2(t)
        x = x3 + t
        return x

class DecoderBilinear(nn.Module):
    def __init__(self, in_chans=384, out_chans=[192, 96], upsample=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans[0], kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),# 384 -> 192
            nn.BatchNorm2d(out_chans[0]),
            nn.GELU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(out_chans[0], out_chans[1], kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 192 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(), # 192 -> 96 = Upsample 2
        )
        self.conv_up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_chans[1], out_chans[1], kernel_size=1, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # 96 -> 96 = Upsample 4
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
        )

    def forward(self, x, x_skip):
        # x 384
        # x_skip [384, 192, 96]
        x = x + x_skip[2]
        x = self.conv_layer1(x)

        x = x + x_skip[1]
        x = self.conv_layer2(x)

        x = x + x_skip[0]
        x = self.conv_up(x)
        return x


class DecoderTranspose2d(nn.Module):
    def __init__(self, in_chans=384, out_chans=[192, 96], upsample=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans[0], kernel_size=2, stride=2),# 384 -> 192
            nn.BatchNorm2d(out_chans[0]),
            nn.GELU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(out_chans[0], out_chans[1], kernel_size=2, stride=2),  # 192 -> 96
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(), # 192 -> 96 = Upsample 2
        )
        self.conv_up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(out_chans[1], out_chans[1], kernel_size=4, stride=4),  # 96 -> 96 = Upsample 4
            nn.BatchNorm2d(out_chans[1]),
            nn.GELU(),
        )

    def forward(self, x, x_skip):
        # x 384
        # x_skip [384, 192, 96]
        x = x + x_skip[2]
        x = self.conv_layer1(x)

        x = x + x_skip[1]
        x = self.conv_layer2(x)

        x = x + x_skip[0]
        x = self.conv_up(x)
        return x





class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)

if __name__ == '__main__':
    tensor = torch.randn(1, 384, 14, 14)
    skip = []
    skip.append(torch.randn(1, 96, 56,56))
    skip.append(torch.randn(1, 192,28,28))
    skip.append(torch.randn(1, 384,14,14))
    decoder = Decoder()
    print(decoder(tensor, skip).shape)

    tensor = torch.randn(1, 16, 224, 224)
    seg = SegmentationHead(16, 3, kernel_size=3)
    print(seg(tensor).shape)