import cv2
import numpy as np
import tensorrt as trt
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import torch
from torchvision import transforms

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img_t = torch.full((3, input_size[0], input_size[1]), 114, dtype=torch.uint8)
    else:
        padded_img_t = torch.full(input_size, 114, dtype=torch.uint8)
    r = min(input_size[0] / img.shape[1], input_size[1] / img.shape[2])
    resized_img_t = transforms.functional.resize(
        img, 
        (int(img.shape[1] * r), int(img.shape[2] * r)), 
        interpolation=transforms.InterpolationMode.BILINEAR
    )
    padded_img_t[:, : resized_img_t.shape[1], : resized_img_t.shape[2]] = resized_img_t
    padded_img_t = padded_img_t.float().contiguous()

    return padded_img_t, r


def infer(input_data, engine_params, cuda_context):
    """Runs inference using the provided engine parameters."""
    host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings, context = engine_params
    cuda_context.push()
    try:
        np.copyto(host_inputs[0], input_data.ravel())
        cuda.memcpy_htod(cuda_inputs[0], host_inputs[0])
        context.execute_v2(bindings)
        # for i in range(len(host_outputs)):
        cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
        reshaped_output = np.reshape(host_outputs[0], (1, 8400, 85))
    finally:
        cuda_context.pop()
    return reshaped_output

def f_infer(img, prepared_engine, cuda_context):
    """Performs inference in a separate thread to avoid threading issues."""
    result = [None]
    def target():
        result[0] = infer(img, prepared_engine, cuda_context)
    infer_thread = threading.Thread(target=target)
    infer_thread.start()
    infer_thread.join()
    return result[0]

def inference_detector(oriImg, engine, cuda_context):
    input_shape = (640,640)
    oriImg = torch.tensor(oriImg).permute(2,0,1)
    img, ratio = preprocess(oriImg, input_shape)
    outputs = f_infer(img, engine, cuda_context)

    predictions = demo_postprocess(outputs, input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        isscore = final_scores>0.3
        iscat = final_cls_inds == 0
        isbbox = [ i and j for (i, j) in zip(isscore, iscat)]
        final_boxes = final_boxes[isbbox]
    else:
        final_boxes = np.array([])

    return final_boxes

