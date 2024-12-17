import os 
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image
import nvtx

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/low_demo.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='exposure', help='Choose from exposure or enhance')
config = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Weights path
exposure_pretrain = r'best_Epoch_exposure.pth'
enhance_pretrain = r'best_Epoch_lol_v1.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Image
img = Image.open(config.file_name)
img = (np.asarray(img)/ 255.0)

model = IAT().to(device)
task = 'enhance'
if task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain))
elif task == 'enhance':
    model.load_state_dict(torch.load(enhance_pretrain))
else:
    warnings.warn('Only could be exposure or enhance')
model.eval()


if img.shape[2] == 4:
    img = img[:,:,:3]
input = torch.from_numpy(img).float().to(device)
input = input.permute(2,0,1).unsqueeze(0)

if config.normalize:    # False
    input = normalize_process(input)

torch.cuda.synchronize()
static_input = input.clone().cuda()

with torch.no_grad():
    for _ in range(3):  # Perform warm-up passes
        _, _, _ = model(static_input)
        
torch.cuda.synchronize()

for _ in range(5):
    _, _ ,enhanced_img = model(input)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
with nvtx.annotate("INFERENCE", color="green"):
    _, _ ,enhanced_img = model(input)
end.record()
torch.cuda.synchronize()
print(f"Execution Time: {(start.elapsed_time(end)/1000):.6f} seconds")
torchvision.utils.save_image(enhanced_img, 'result.png')
