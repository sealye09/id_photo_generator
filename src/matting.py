import math

import cv2
import numpy as np
import onnxruntime

from src.enums import BackgroundColor, MattingModel


# Get x_scale_factor & y_scale_factor to resize image
def get_scale_factor(im_h, im_w, ref_size):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor


def image_preprocess(im, ref_size=512):
    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype("float32")

    return im, im_w, im_h


def image_postprocess(matte, im_w, im_h):
    # refine matte
    matte = (np.squeeze(matte) * 255).astype("uint8")
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)
    return matte


def onnx_inference(model_path, im):
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})
    return result


class Matte:
    def __init__(
        self,
        model_path=MattingModel.MODNET.value,
        ref_size=512,
        background_color=BackgroundColor.BLUE.value,
    ):
        self.model_path = model_path
        self.ref_size = ref_size
        self.background_color = background_color

    def matting(self, image: np.ndarray):
        im_pre, im_w, im_h = image_preprocess(image, self.ref_size)
        result = onnx_inference(self.model_path, im_pre)
        background = image_postprocess(result[0], im_w, im_h)
        return background

    def compose(self, foreground, background):
        im_h, im_w, im_c = foreground.shape
        background = cv2.resize(
            background, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA
        )
        b, g, r = cv2.split(foreground)
        result = cv2.merge([r, g, b, background])

        return result

    def compose_with_background(self, foreground: np.ndarray, background: np.ndarray):
        """
        图像融合：合成图 = 前景*alpha+背景*(1-alpha)
        :param foreground: RGB图像(uint8)
        :param background: 单通道的alpha图像(uint8)
        :return: 返回与背景合成的图像
        """
        if isinstance(self.background_color, tuple) or isinstance(
            self.background_color, list
        ):
            self.background_color = np.ones_like(foreground, dtype=np.uint8) * np.array(
                self.background_color, dtype=np.uint8
            )

        if len(background.shape) == 2:
            background = background[:, :, np.newaxis]
        if background.dtype == np.uint8:
            background = np.asarray(background / 255.0, dtype=np.float32)

        sh, sw, d = foreground.shape
        bh, bw, d = self.background_color.shape
        ratio = [sw / bw, sh / bh]
        ratio = max(ratio)
        if ratio > 1:
            self.background_color = cv2.resize(
                self.background_color,
                dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
                interpolation=cv2.INTER_AREA,
            )

        # Convert arrays to the same data type
        foreground = foreground.astype(np.float32)
        background = background.astype(np.float32)
        self.background_color = self.background_color.astype(np.float32)

        # Perform element-wise multiplication
        result = foreground * background + self.background_color * (
            np.array([1], dtype=background.dtype) - background
        )
        result = result.astype(np.uint8)

        return result
