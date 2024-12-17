import cv2
import numpy as np
from PIL import Image


def calculate_brightness(image):
    """
    Calculate the brightness level of an image.
    Args:
        image (numpy.ndarray): Input image in range [0, 1].
    Returns:
        float: Brightness level (0 for dark, 1 for bright).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    brightness = np.mean(gray)  # Calculate average brightness
    return brightness

file_name = "con.jpeg"
img = Image.open(file_name)
img = (np.asarray(img)/ 255.0)
img = img.astype(np.float32)


brightness = calculate_brightness(img)
print(brightness)