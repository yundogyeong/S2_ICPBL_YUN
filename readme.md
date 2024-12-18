# S2 공연 환경에 적용 가능한 human pose estimation 

## 프로젝트 설명
공연장환경(어두운 조명, 다중 군중 등)에 대응 가능한 human pose estimation 기술
## 환경설정

TensorRT Docker 환경에서 모델을 실행 가능. 아래 링크를 참고하여 TensorRT Docker 이미지와 사용법 확인:

[TensorRT Docker 환경 설정 가이드](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/)

---
## 0. tensorRT 모델 준비

이미지 복원 모델
```bash
# enhance
trtexec --onnx="enhance.onnx" --saveEngine="enhance.trt" --fp16
mv enhance.trt Illumination-Adaptive-Transformer/IAT_enhance/

# exposure
trtexec --onnx="exposure.onnx" --saveEngine="exposure.trt" --fp16
mv exposure.trt Illumination-Adaptive-Transformer/IAT_enhance/
```

포즈 추정 모델
```bash
# detection
trtexec --onnx="yolox_l.onnx" --saveEngine="yolo.trt" --fp16
mv yolo.trt pose/ControlNet-v1-1-nightly/

# pose
trtexec --onnx="dw-ll_ucoco_384.onnx" --saveEngine="pose.trt" --fp16
mv pose.trt pose/ControlNet-v1-1-nightly/
```
---
## 1. 이미지 복원 모델 실행

```bash
cd Illumination-Adaptive-Transformer/IAT_enhance
python3 trt_demo.py --input "이미지 경로"
```

## 2. 포즈 추정 모델 실행

```bash
cd pose/ControlNet-v1-1-nightly
python3 dwpose_infer_example.py --input "이미지 경로"
```

