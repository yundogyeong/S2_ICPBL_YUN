import argparse
import cv2
import matplotlib.pyplot as plt
import time
import torch
from annotator.dwpose import DWposeDetector

def main(input_image, output_image):
    pose = DWposeDetector()
    oriImg = cv2.imread(input_image)  # B,G,R order
    
    for _ in range(5):
        _ = pose(oriImg)
        
    torch.cuda.synchronize()
    start_time = time.time()
    out = pose(oriImg)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Inference Time: {end_time - start_time:.4f} seconds")
    
    plt.imsave(output_image, out)
    print(f"Result saved to: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DWposeDetector Inference")
    parser.add_argument("--input", type=str, default='test_imgs/pose1.png',  help="Path to the input image")
    parser.add_argument("--output", type=str, default='result.png', help="Path to save the output image")
    
    args = parser.parse_args()
    main(args.input, args.output)
