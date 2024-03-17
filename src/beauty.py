import cv2
import numpy as np


class Beauty:
    def __init__(self, image):
        self.image = image

    def beautify_skin(
        self, bilateral_d, bilateral_sigma_color, bilateral_sigma_space, gamma
    ):
        """
        美肤处理
        :param image_path: 输入图像的路径
        :param bilateral_d: 双边滤波的直径
        :param bilateral_sigma_color: 双边滤波的颜色标准差
        :param bilateral_sigma_space: 双边滤波的空间标准差
        :param gamma: 幂函数增强的参数
        :return: 处理后的图像
        """
        # Convert the image to YCrCb color space
        image_ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)

        # Split the Y, Cr, and Cb channels
        y, cr, cb = cv2.split(image_ycrcb)

        # Apply bilateral filter to the Y channel
        y = cv2.bilateralFilter(
            y, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
        )

        # Apply power function enhancement to the Y channel
        y = np.clip((y.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(
            np.uint8
        )

        # Merge the channels back
        image_ycrcb = cv2.merge([y, cr, cb])

        # Convert the image back to BGR color space
        beautified_image = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

        return beautified_image
