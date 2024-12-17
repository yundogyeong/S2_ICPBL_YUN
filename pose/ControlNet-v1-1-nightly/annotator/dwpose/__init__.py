# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import tensorrt as trt
from . import util
from .wholebody import Wholebody

def load_engine(engine_file_path):
    """Loads a TensorRT engine from the specified file path."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self):
        self.yolo_host_inputs, self.yolo_cuda_inputs, self.yolo_host_outputs, self.yolo_cuda_outputs, self.yolo_bindings = [], [], [], [], []
        self.pose_host_inputs, self.pose_cuda_inputs, self.pose_host_outputs, self.pose_cuda_outputs, self.pose_bindings = [], [], [], [], []

        self.pose_estimation = Wholebody()
        self.yolo_engine = load_engine("/workspace/DWPose/ControlNet-v1-1-nightly/yolo.trt")
        self.pose_engine = load_engine("/workspace/DWPose/ControlNet-v1-1-nightly/pose.trt")
        
        self.yolo_prepared_engine = self.prepare_yolo_trt_engine(self.yolo_engine)
        self.pose_prepared_engine = self.prepare_pose_trt_engine(self.pose_engine)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg,  self.yolo_prepared_engine, self.pose_prepared_engine)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)
        
    def prepare_yolo_trt_engine(self, engine):
        """Prepares the engine for inference by setting up buffers and bindings."""
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.pose_bindings.append(int(cuda_mem))
            
            if engine.binding_is_input(binding):
                self.pose_host_inputs.append(host_mem)
                self.pose_cuda_inputs.append(cuda_mem)
            else:
                self.pose_host_outputs.append(host_mem)
                self.pose_cuda_outputs.append(cuda_mem)

        return self.pose_host_inputs, self.pose_cuda_inputs, self.pose_host_outputs, self.pose_cuda_outputs, self.pose_bindings, engine.create_execution_context()

    def prepare_pose_trt_engine(self, engine):
        """Prepares the engine for inference by setting up buffers and bindings."""
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.yolo_bindings.append(int(cuda_mem))
            
            if engine.binding_is_input(binding):
                self.yolo_host_inputs.append(host_mem)
                self.yolo_cuda_inputs.append(cuda_mem)
            else:
                self.yolo_host_outputs.append(host_mem)
                self.yolo_cuda_outputs.append(cuda_mem)

        return self.yolo_host_inputs, self.yolo_cuda_inputs, self.yolo_host_outputs, self.yolo_cuda_outputs, self.yolo_bindings, engine.create_execution_context()

