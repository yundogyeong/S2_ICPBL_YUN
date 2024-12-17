import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import Infer

class Wholebody:
    def __init__(self):
        device = 'cuda:0'
        self.cuda_context = cuda.Device(0).make_context()
        self.inference_pose = Infer()

    
    def __call__(self, oriImg, yolo_engine, pose_engine):
        det_result = inference_detector(oriImg, yolo_engine, self.cuda_context)
        # torch.cuda.synchronize()
        keypoints, scores = self.inference_pose.inference_pose(det_result, oriImg, pose_engine)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores
    
    def __del__(self):
        self.cuda_context.pop()


