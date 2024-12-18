import os
import sys
import threading
import argparse
import warnings

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import Normalize, Resize
from PIL import Image
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from BN_model import BrightnessClassifier
cuda.init()
# CUDA context and buffer initialization
host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings = [], [], [], [], []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_engine(engine_file_path):
    """Loads a TensorRT engine from the specified file path."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)

def prepare_trt_engine(engine):
    """Prepares the engine for inference by setting up buffers and bindings."""
    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    return host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, engine.create_execution_context()

def infer(input_data, engine_params, cuda_context):
    """Runs inference using the provided engine parameters."""
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, context = engine_params
    cuda_context.push()
    try:
        np.copyto(host_inputs[0], input_data.ravel())
        cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
        context.execute_v2(bindings)
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh(host_outputs[i], cuda_outputs[i])
    finally:
        cuda_context.pop()
    return host_outputs

def f_infer(input_data, prepared_engine, cuda_context):
    """Performs inference in a separate thread to avoid threading issues."""
    result = [None]
    def target():
        result[0] = infer(input_data, prepared_engine, cuda_context)
    infer_thread = threading.Thread(target=target)
    infer_thread.start()
    infer_thread.join()
    return result[0]

def test_single_image(model, input, transform, device):
    model.eval()
    model = model.to(device).half() 

    if transform:
        image = transform(input)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  
        _, predicted = torch.max(probabilities, 1)  
        print(predicted)

    labels = {0: "어두움", 1: "중간", 2: "밝음"}
    predicted_label = labels[predicted.item()]

    print(f"Predicted Brightness: {predicted_label}")
    return predicted

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/low_demo.jpg')
parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--task', type=str, default='exposure', help='Choose from exposure or enhance')
config = parser.parse_args()

exposure_pretrain = r'best_Epoch_exposure.pth'
enhance_pretrain = r'best_Epoch_lol_v1.pth'


# Pre-process input image
img = Image.open(config.file_name)
img = np.asarray(img) / 255.0
if img.shape[2] == 4:  # If the image has an alpha channel, remove it
    img = img[:, :, :3]

input = torch.from_numpy(img).float().to(device)
input = input.permute(2,0,1).unsqueeze(0).half()
brightness_model = BrightnessClassifier()
brightness_model.load_state_dict(torch.load("brightness_classifier.pth"))
transform = transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale()])
predicted_label = test_single_image(brightness_model, input, transform, device)
input = input.cpu()
cuda_context = cuda.Device(0).make_context()

if predicted_label == 0:
    # Load engine and prepare for inference
    engine = load_engine("enhance.trt")
    prepared_engine = prepare_trt_engine(engine)
elif predicted_label == 1:
    sys.exit(0)
else:
    engine = load_engine("exposure.trt")
    prepared_engine = prepare_trt_engine(engine)


torch.cuda.synchronize()
for _ in range(5):
    _, _, enhanced_img = f_infer(input, prepared_engine, cuda_context)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
_, _, enhanced_img = f_infer(input, prepared_engine, cuda_context)
end.record()
torch.cuda.synchronize()
reshaped_output = np.reshape(enhanced_img, (1, 3, 338, 506))
output_tensor = torch.from_numpy(reshaped_output).to(device="cuda", dtype=torch.float16)
print(f"Graph Execution Time: {(start.elapsed_time(end)/1000):.6f} seconds")


torchvision.utils.save_image(output_tensor, 'output_image.jpeg')
cuda_context.pop()