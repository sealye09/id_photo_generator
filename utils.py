import cv2
import numpy as np


def scaleImage2float(image):
    """
    Scale to Images of type float between -1 and 1.
    """
    image = image.astype(np.float32)  # Convert the image to float
    image = (image / 127.5) - 1  # Scale the image from [0,255] to [-1,1]

    return image


def scaleImage2uint8(image):
    """
    Scale to Images of type uint8 between 0 and 255.
    """
    image = (image + 1) * 127.5  # Scale the image from [-1,1] to [0,255]
    image = image.astype(np.uint8)  # Convert the image to uint8

    return image


def convert2BGR(image):
    """
    Convert the image to BGR format.
    """
    # 1 channel
    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 2 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 4 channels
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # 3 channels (RGB)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
