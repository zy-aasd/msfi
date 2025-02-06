import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model.launch_seg import launch_seg as ViT_seg
from utils.grad_utils import GradCAM, show_cam_on_image, center_crop_img
from matplotlib import pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),
])
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../../data/Synapse/',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='../../data/Synapse/list', help='list dir')
parser.add_argument('--output_dir', default='../model/ip-synapse', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving predictions as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument(
    "--opts",
    help="Modify configs options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped datasets instead of folder datasets')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the datasets into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
snapshot = os.path.join(args.output_dir,
                        'epoch_synapse_' + str(args.max_epochs - 1) + "_lr_" + str(args.base_lr) + "_seed" +
                        str(args.seed) + '.pth')
model = ViT_seg(num_classes=9, pretrain='res50')
print(model)


msg = model.load_state_dict(torch.load(snapshot))
target_layers = [model.model.segmentationHead[0]]
# .segmentationHead[0]
# .decoder.conv_layer1[0]
# .encoder.drop
img_path = 'datasets/image/case0025_67.png'
img = Image.open(img_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
img_tensor = transform(img)
# [C, H, W] -> [N, C, H, W]
img_tensor = torch.unsqueeze(img_tensor, dim=0)
print(img_tensor.shape)

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
visualization=[]
for idx in range(9):
    if idx != 0:
        target_category = idx
        # target_category = 254  # pug, pug-dog
        grayscale_cam = cam(input_tensor=img_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization.append(show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                                      grayscale_cam,
                                                      use_rgb=True))
        plt.imshow(visualization[idx-1])
        cv2.imwrite(f'./heatmap/case0025_67_{idx}.png', visualization[idx-1])

