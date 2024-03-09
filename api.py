import base64
from typing import Annotated

import cv2 as cv
import numpy as np
from fastapi import Body, FastAPI

from beauty import Beauty
from enums import BackgroundColor, MattingModel, PhotoExtension, PhotoSize
from matting import inference
from photo_cropper import PhotoCropper


def ID_photo_generator(
    image: cv.typing.MatLike,
    background_color=BackgroundColor.BLUE.value,
    photo_size=PhotoSize.SMALL.value,
    extension=PhotoExtension.PNG.value,
):
    """
    得到裁剪完成的证件照
    :param image: 输入图像
    :param background_color: 背景颜色
    :param photo_size: 证件照尺寸
    """
    if image is None:
        print("Image is None")
        return None

    # check background color by name
    if not BackgroundColor(background_color):
        print("Invalid background color")
        return None

    # check photo size by name
    if not PhotoSize(photo_size):
        print("Invalid photo size")
        return None

    # 人像分割结果和合成图像
    (matte, compose_im, compose_im_with_bg) = inference(
        image=image,
        model_path=MattingModel.MODNET.value,
        background_color=background_color,
    )

    # 获取人脸区域，裁剪证件照
    cropper = PhotoCropper(compose_im_with_bg, matte)
    cropped_img = cropper.crop()

    # check if the cropped image is None
    if cropped_img is None:
        print("Cropped went wrong. Please try again.")
        return None

    # 美颜处理
    beauty = Beauty(cropped_img)
    beauty_image = beauty.beautify_skin(5, 12, 12, 1.2)

    # 修改证件照尺寸。默认为1寸
    cropped_img = cv.resize(cropped_img, photo_size, interpolation=cv.INTER_AREA)

    # 格式 png jpg
    if extension == PhotoExtension.JPG.value:
        beauty_image = cv.imencode(".jpg", np.array(beauty_image))[1]
        beauty_image = beauty_image.tobytes()
    else:
        beauty_image = cv.imencode(".png", np.array(beauty_image))[1]
        beauty_image = beauty_image.tobytes()

    return matte, compose_im_with_bg, cropped_img, beauty_image


def numpy_2_base64(img: np.ndarray):
    """Convert numpy array to base64 string

    Args:
        img (np.ndarray): input image

    Returns:
        str: base64 string
    """
    retval, buffer = cv.imencode(".png", img)
    base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return base64_image


def base64_2_numpy(base64_str: str):
    """Convert base64 string to numpy array

    Args:
        base64_str (str): base64 string

    Returns:
        np.ndarray: output image
    """
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    return img


app = FastAPI()


@app.get("/api/hello")
def read_root():
    return {"Hello": "World"}


# 证件照智能制作接口
@app.post("/api/id-photo")
async def id_photo_inference(
    image: Annotated[str, Body(embed=True)],
    background_color: Annotated[tuple, Body(embed=True)] = BackgroundColor.BLUE.value,
    size: Annotated[tuple, Body(embed=True)] = PhotoSize.SMALL.value,
    extension: Annotated[str, Body(embed=True)] = PhotoExtension.PNG.value,
):
    # check size background_color extension by value
    if not PhotoSize(size):
        result_message = {"code": 400, "data": None, "message": "Invalid photo size"}
        return result_message
    if not BackgroundColor(background_color):
        result_message = {
            "code": 400,
            "data": None,
            "message": "Invalid background color",
        }
        return result_message
    if not PhotoExtension(extension):
        result_message = {
            "code": 400,
            "data": None,
            "message": "Invalid photo extension",
        }
        return result_message

    input_image = base64_2_numpy(image)

    res = ID_photo_generator(input_image, background_color, size, extension)

    if res is None:
        result_message = {"code": 500, "data": None, "message": "failed"}
        return result_message

    beauty_image = res[3]
    if beauty_image is None:
        result_message = {"code": 500, "data": None, "message": "failed"}
        return result_message

    # Convert to base64
    res_image = numpy_2_base64(np.frombuffer(beauty_image, np.uint8))

    result_message = {"code": 200, "data": res_image, "message": "success"}
    return result_message


if __name__ == "__main__":
    import uvicorn

    # 在8080端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=3030)
