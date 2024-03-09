import math

import cv2
import numpy as np
import onnxruntime


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


def compose_image(image, matte):
    im_h, im_w, im_c = image.shape
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(image)
    result = cv2.merge([r, g, b, matte])

    return result


def image_fusion(image: np.ndarray, alpha: np.ndarray, bg_img=(255, 0, 0)):
    """
    图像融合：合成图 = 前景*alpha+背景*(1-alpha)
    :param image: RGB图像(uint8)
    :param alpha: 单通道的alpha图像(uint8)
    :param bg_img: 背景图像,可以是任意的分辨率图像，也可以指定指定纯色的背景
    :return: 返回与背景合成的图像
    """
    if isinstance(bg_img, tuple) or isinstance(bg_img, list):
        bg_img = np.ones_like(image, dtype=np.uint8) * np.array(bg_img, dtype=np.uint8)

    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
    if alpha.dtype == np.uint8:
        alpha = np.asarray(alpha / 255.0, dtype=np.float32)

    sh, sw, d = image.shape
    bh, bw, d = bg_img.shape
    ratio = [sw / bw, sh / bh]
    ratio = max(ratio)
    if ratio > 1:
        bg_img = cv2.resize(
            bg_img,
            dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
            interpolation=cv2.INTER_AREA,
        )

    # Convert arrays to the same data type
    image = image.astype(np.float32)
    alpha = alpha.astype(np.float32)
    bg_img = bg_img.astype(np.float32)

    # Perform element-wise multiplication
    result = image * alpha + bg_img * (np.array([1], dtype=alpha.dtype) - alpha)
    result = result.astype(np.uint8)

    return result


def onnx_inference(model_path, im):
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})
    return result


def inference(image, model_path, ref_size=512, background_color=(0, 0, 255)):
    im_pre, im_w, im_h = image_preprocess(image, ref_size)
    result = onnx_inference(model_path, im_pre)
    matte = image_postprocess(result[0], im_w, im_h)
    compose_im = compose_image(image, matte)

    compose_im_with_bg = image_fusion(image, matte, background_color)

    return matte, compose_im, compose_im_with_bg
