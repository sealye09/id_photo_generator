from enum import Enum


class BackgroundColor(Enum):
    BLUE = (67, 142, 219)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


class PhotoSize(Enum):
    # 1寸
    SMALL = (295, 413)
    # 2寸
    LARGE = (413, 626)


class PhotoExtension(Enum):
    PNG = "png"
    JPG = "jpg"


class MattingModel(Enum):
    MODNET = "public/models/matting/modnet_photographic_portrait_matting.onnx"


class FaceDetectorModel(Enum):
    CAFFE_BIN = (
        "public/models/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )
    CAFFE_CONFIG = "public/models/face_detector/deploy.prototxt"


def is_valid_background_color(color):
    return color in [c.value for c in BackgroundColor]


def is_valid_photo_size(size):
    return size in [s.value for s in PhotoSize]
