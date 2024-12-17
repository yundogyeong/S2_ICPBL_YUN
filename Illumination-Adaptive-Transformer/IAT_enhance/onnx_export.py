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

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/low_demo.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='enhance', help='Choose from exposure or enhance')
config = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Weights path
exposure_pretrain = r'best_Epoch_exposure.pth'
enhance_pretrain = r'best_Epoch_lol_v1.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Pre-train Weights
model = IAT().to(device)
if config.task == 'exposure':
    model.load_state_dict(torch.load(exposure_pretrain))
elif config.task == 'enhance':
    model.load_state_dict(torch.load(enhance_pretrain))
else:
    warnings.warn('Only could be exposure or enhance')
model.eval().half()

onnx_export = True
## Load Image
img = Image.open(config.file_name)
img = (np.asarray(img)/ 255.0)
if img.shape[2] == 4:
    img = img[:,:,:3]
input = torch.from_numpy(img).float().to(device)
input = input.permute(2,0,1).unsqueeze(0).half()
if config.normalize:    # False
    input = normalize_process(input)

if onnx_export:
    onnx_path = "IAT_model.onnx"  # Define ONNX file path
    dummy_input = input.clone()  # Use the same input format as during inference
    torch.onnx.export(
        model,  # Model to export
        dummy_input,  # Dummy input tensor
        onnx_path,  # Output ONNX file
        export_params=True,  # Store the trained parameters in the ONNX file
        opset_version=17,  # ONNX opset version
        do_constant_folding=True,  # Optimize the graph
        input_names=["input"],  # Input tensor name
        output_names=["output"],  # Output tensor name
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Dynamic batch size
    )
    print(f"Model has been exported to {onnx_path}")
