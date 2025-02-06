import numpy as np
import thop
import torch
import torchvision
from thop import profile
from torch import nn
from model.PatchMerging import PatchMerging
from model.PatchEmbed import PatchEmbed
from model.launch_seg import launch_seg as seg_net
print(torch.cuda.is_available())
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=F._interpolation_modes_from_int(0)),
])
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'
cnn_model = torchvision.models.resnet50(pretrained=True)
resnet_layers = nn.ModuleList(cnn_model.children())[:12]
print(resnet_layers)
#
# cnn_model = torchvision.models.resnet34(pretrain=True) # False-21.02G 22.3316M , True-
# resnet_layers = nn.ModuleList(cnn_model.children())[:12]
# print(resnet_layers)
# res50:0~4 = 3->256, 5=256->128->512 6=512->256->1024
# res34:0~4:3-》64，  5:64->128 6:128->256  7:256->512
# res101:0~4 = 3->256, 5=256->128->512 6=512->256->1024
# #
model = seg_net(num_classes=9, pretrain='res50')
dummy_input = torch.rand(1, 3, 224, 224) # 1 batch
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.4fG, params: %.4fM' % (flops / 1000000000, params / 1000000))

# res34= flops: 12.2309G, params: 42.8632M
# res50= flops: 20.7624G, params: 18.7473M
model = seg_net(num_classes=3, pretrain='res50')
print(model)