import os
import cv2 as cv
import numpy as np
import gradio as gr

from matting import inference
from photo_cropper import PhotoCropper

# MOD_NET_MODEL_PATH
MATTING_MODEL_PATH = "public/models/modnet_photographic_portrait_matting.onnx"

# FACE_DETECTOR_MODEL_BIN
FACE_DETECTOR_MODEL_BIN = (
    "public/models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
# FACE_DETECTOR_CONFIG_TEXT
FACE_DETECTOR_CONFIG_TEXT = "public/models/face_detector/deploy.prototxt"


def matting(image):
    """
    :param image: 输入图像
    :return: 人像分割结果和合成图像
    """
    (matte, compose_im, compose_im2) = inference(
        image=image, model_path=MATTING_MODEL_PATH
    )

    return (matte, compose_im, compose_im2)


def get_face_info(image):
    """
    获取图像中的人脸信息
    :param image: 输入图像
    :return: 人脸信息
    :return[0]: 人脸区域
    """

    # 加载人脸检测模型
    face_detector = cv.FaceDetectorYN.create(
        model="public/models/face_detector/yunet_n_320_320.onnx",
        config="",
        input_size=(image.shape[1], image.shape[0]),
    )

    # convert the image to 3 channels
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    # 人脸检测
    detections = face_detector.detect(image)

    result = np.array([]) if detections[1] is None else detections[1]

    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255),  # left mouth corner
    ]
    text_color = (0, 0, 255)
    box_color = (0, 255, 0)

    bbox = None
    landmarks = None
    for det in result:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            box_color,
            2,
        )

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(image, landmark, 2, landmark_color[idx], 2)

    return bbox, landmarks


def predict(image):
    """
    :param image: 输入图像
    """
    if image is None:
        return None, None, None

    start_time = cv.getTickCount()
    (matte, compose_im, compose_im_with_bg) = matting(image)

    cv.cvtColor(compose_im_with_bg, cv.COLOR_BGR2RGB)

    cropper = PhotoCropper(compose_im_with_bg, matte)
    cropped_img = cropper.crop()

    # 计算时间
    time = (cv.getTickCount() - start_time) / cv.getTickFrequency()

    print("time: %.2f s" % time, flush=True)

    return matte, compose_im_with_bg, cropped_img


if __name__ == "__main__":
    # 测试图像为image中名称包含image的图像
    example_image_path = "public/images"

    examples = [
        os.path.join(example_image_path, i)
        for i in os.listdir(example_image_path)
        if "image" in i
    ]

    # examples按照名称排序
    examples.sort(key=lambda x: x.split("/")[-1].split(".")[0].split("_")[-1])

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Image", height=512),
        outputs=[
            gr.Image(type="pil", label="Matte"),
            gr.Image(type="pil", label="Compose Image"),
            gr.Image(type="pil", label="Face Area"),
        ],
        title="person-modnet",
        examples=examples,
        theme=gr.themes.Base(),
    )
    interface.launch(share=False)
