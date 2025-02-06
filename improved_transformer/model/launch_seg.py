import copy

import torch
from torch import nn
from .Model import Model

class launch_seg(nn.Module):
    def __init__(self, img_size=224, num_classes=3, pretrain='res50'):
        super(launch_seg, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = Model(img_size=img_size, num_classes=num_classes, pretrain=pretrain)
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.model(x)
        return logits

