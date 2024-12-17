from annotator.dwpose import DWposeDetector

if __name__ == "__main__":
    pose = DWposeDetector()
    import cv2
    test_image = 'test_imgs/pose1.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    import matplotlib.pyplot as plt
    import time
    import torch
    for _ in range(5):
        out = pose(oriImg)
    torch.cuda.synchronize()
    start_time = time.time()
    out = pose(oriImg)
    torch.cuda.synchronize()
    end_time = time.time()
    print(end_time - start_time)
    plt.imsave('result.jpg', out)
