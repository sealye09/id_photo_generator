import cv2
import numpy as np

from src.enums import FaceDetectorModel


class PhotoCropper:
    """
    证件照裁剪类
    输入原图和 matting 后的图像，输出证件照
    1. 通过人脸检测模型检测人脸区域
    2. 通过 matting 后的图像检测最高
    3. 根据人脸的关键点信息检测肩部位置，计算证件照的底部位置
    4. 根据比例来调整相片的区域
    5. 关键点调整人像的位置，居中
    6. 裁剪证件照，超过的区域使用边缘信息填充
    """

    def __init__(self, input_image, matte_image):
        self.face_detector = cv2.dnn.readNetFromCaffe(
            FaceDetectorModel.CAFFE_CONFIG.value,
            FaceDetectorModel.CAFFE_BIN.value,
        )
        self.face_on_photo = 0.7
        self.photo_ratio = 5 / 7.0

        # face area
        self.face_top = 0
        self.face_bottom = 0
        self.face_left = 0
        self.face_right = 0

        # photo area
        self.photo_top = 0
        self.photo_bottom = 0
        self.photo_left = 0
        self.photo_right = 0

        # input image area
        self.input_image_top = 0
        self.input_image_bottom = 0
        self.input_image_left = 0
        self.input_image_right = 0

        self.input_image: cv2.typing.MatLike = input_image
        self.matte_image: cv2.typing.MatLike = matte_image

        self.blob: cv2.typing.MatLike | None = None
        self.detections: cv2.typing.MatLike | None = None

        h = self.input_image.shape[0]
        w = self.input_image.shape[1]
        # input image area
        self.input_image_top = 0
        self.input_image_bottom = h
        self.input_image_left = 0
        self.input_image_right = w

    def crop(self):
        """
        裁剪证件照
        """
        self._preprocess_input()
        self._detect_faces()

        self._detect_top()
        self._detect_bottom()
        self._detect_x(self.photo_top, self.photo_bottom)

        self._fix_photo_area()

        # 裁剪证件照
        photo = self.input_image[
            self.photo_top : self.photo_bottom, self.photo_left : self.photo_right
        ]

        return photo

    def get_photo_area(self):
        """
        获取证件照区域
        """
        return (
            self.photo_left,
            self.photo_top,
            self.photo_right,
            self.photo_bottom,
        )

    def _preprocess_input(self):
        """
        输入图像预处理
        """
        # convert the image to 3 channels RGB
        if len(self.input_image.shape) == 2:
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_GRAY2RGB)
        if self.input_image.shape[2] == 4:
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGRA2RGB)

    def _detect_faces(self):
        self.blob = cv2.dnn.blobFromImage(
            cv2.resize(self.input_image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )

        self.face_detector.setInput(self.blob)
        detections = self.face_detector.forward()

        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # confidence > 0.6
        if confidence > 0.6:
            self.detections = detections[0, 0, i, 3:7]
            (h, w) = self.input_image.shape[:2]
            box = self.detections * np.array([w, h, w, h])
            (self.face_left, self.face_top, self.face_right, self.face_bottom) = (
                box.astype("int")
            )

            return detections[0, 0, i, 3:7]

        self.detections = None

    def _fix_photo_area(self):
        # 将人脸区域重置成正方形，以长的一边为准，中心对齐
        face_width = self.face_right - self.face_left
        face_height = self.face_bottom - self.face_top
        face_center_x = (self.face_left + self.face_right) / 2
        face_center_y = (self.face_top + self.face_bottom) / 2
        if face_width > face_height:
            self.face_top = int(face_center_y - face_width / 2)
            self.face_bottom = int(face_center_y + face_width / 2)
        else:
            self.face_left = int(face_center_x - face_height / 2)
            self.face_right = int(face_center_x + face_height / 2)

        # 扩大证件照区域
        # 1. 证件照区域的高度为人脸区域的比例 1：0.675
        # 2. 证件照区域的长宽比例为 5:7
        # 3. 证件照区域的中心与人脸区域的中心重合
        photo_height = int((self.face_bottom - self.face_top) / self.face_on_photo)
        photo_width = int(photo_height * self.photo_ratio)
        photo_center_x = (self.face_left + self.face_right) / 2
        photo_center_y = (self.face_top + self.face_bottom) / 2

        new_photo_width = photo_width / self.face_on_photo
        new_photo_height = new_photo_width / self.photo_ratio

        # 调整证件照区域的位置, 使证件照区域的中心与人脸区域的中心重合, 同时设置证件照区域
        self.photo_left = int(photo_center_x - new_photo_width / 2)
        self.photo_right = int(photo_center_x + new_photo_width / 2)
        self.photo_top = int(photo_center_y - new_photo_height / 2)
        self.photo_bottom = int(photo_center_y + new_photo_height / 2)

        # 4. 调整证件照区域的 y 轴位置, 以 head_top 为标准
        # head_top 在 photo_top 下方，照片上方区域可能太大
        if self.head_top > self.photo_top:
            y = self.head_top - self.photo_top
            exchange = max(int(y / 2), 20)
            if y >= 20:
                self.photo_top += exchange
                self.photo_bottom += exchange

    def _detect_top(self):
        """
        通过 matting 后的图像检测最高点
        """
        top = 0
        for i in range(self.matte_image.shape[0]):
            if np.max(self.matte_image[i]) > 0.5:
                top = i
                break
        self.photo_top = top
        self.head_top = top
        return top

    def _detect_x(self, top, bottom, threshold=5):
        """
        在 top - bottom 之间检测 left 和 right
        """
        left = 0
        right = 0

        for i in range(self.matte_image.shape[1]):
            if np.max(self.matte_image[top:bottom, i].astype(float)) > threshold:
                left = i
                break

        for i in range(self.matte_image.shape[1] - 1, 0, -1):
            if np.max(self.matte_image[top:bottom, i].astype(float)) > threshold:
                right = i
                break

        self.photo_left = left
        self.photo_right = right
        return left, right

    def _detect_bottom(self):
        """
        检测证件照的底部位置，大概为肩膀的位置
        根据比例来调整相片的区域
        """
        bottom = 0
        if self.detections is not None:
            (h, w) = self.input_image.shape[:2]
            box = self.detections * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            center_x = (startX + endX) / 2
            center_y = (startY + endY) / 2
            len = max(endX - startX, endY - startY)
            bottom = int(center_y + len * self.face_on_photo)

        # 是否越界
        bottom = min(bottom, self.input_image_bottom)
        self.photo_bottom = bottom

    def draw(self):
        """
        绘制检测矩形
        根据已有的photo area
        """
        photo = self.input_image.copy()
        cv2.rectangle(
            photo,
            (self.face_left, self.face_top),
            (self.face_right, self.face_bottom),
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            photo,
            (self.photo_left, self.photo_top),
            (self.photo_right, self.photo_bottom),
            (0, 255, 0),
            2,
        )

        return photo


def edge_blur(image, kernel_size, low_threshold, high_threshold):
    """
    对图像边缘进行模糊处理
    :param image: 输入的图像
    :param kernel_size: 高斯模糊的核大小，应为奇数
    :param low_threshold: Canny边缘检测的低阈值
    :param high_threshold: Canny边缘检测的高阈值
    :return: 处理后的图像
    """
    original_image = image.copy()

    # Perform Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Create an empty kernel
    kernel = np.ones((3, 3), np.uint8)

    # Dilate the edges to create a mask
    dilated_edges = cv2.dilate(edges, kernel)

    # Create a mask with Gaussian blur
    mask = cv2.GaussianBlur(dilated_edges, (kernel_size, kernel_size), 0)

    # Convert the mask to float32
    mask = mask.astype(np.float32) / 255

    # Blend the image and the mask
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blurred = cv2.multiply(original_image.astype(np.float32), mask_3d)

    # Convert the result back to uint8
    blurred = blurred.astype(np.uint8)

    return blurred


# if __name__ == "__main__":
#     from matting import matting

#     start = cv2.getTickCount()
#     image = cv2.imread("PPM-100/image/3104502752_cb935c1f0b_o.jpg")

#     (matte, compose_im, compose_im_with_bg) = matting(image)

#     time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
#     print("time1: %.2f s" % time, flush=True)

#     photo_cropper = PhotoCropper(image, matte)
#     photo_cropper.crop()
#     photo = photo_cropper.draw()

#     # 颜色空间转换
#     photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
#     time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
#     print("time2: %.2f s" % time, flush=True)
#     plt.imshow(photo)
#     plt.show()
#     print("done")
