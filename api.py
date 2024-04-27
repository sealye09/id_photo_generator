import base64
import mimetypes
from typing import Annotated

import cv2 as cv
import numpy as np
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.beauty import Beauty
from src.enums import BackgroundColor, PhotoExtension, PhotoSize
from src.matting import Matte
from src.photo_cropper import PhotoCropper


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

    # 人像分割结果和合成图像
    matter = Matte(background_color=background_color)
    matte = matter.matting(image)
    compose_im_with_bg = matter.compose_with_background(image, matte)

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


def parse_base64(base64_str: str):
    if base64_str.startswith("data:image"):
        ext = "png"
        base64_str, data = base64_str.split(",")  # separate header and data
        mime_type = base64_str.split(";")[0].split(":")
        if len(mime_type) > 1:
            mime_type = mime_type[1]
            extension = mimetypes.guess_extension(mime_type)
            if extension is not None and extension.startswith("."):
                ext = extension[1:]

        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)

        return img, ext
    else:
        data = base64_str.split(",")[1]
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return img, "png"


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/hello")
def read_root():
    return {"Hello": "World"}


# 证件照智能制作接口
@app.post("/api/id-photo")
async def id_photo_inference(
    image: Annotated[str, Body(embed=True)],
    background_color: Annotated[
        tuple | str, Body(embed=True)
    ] = BackgroundColor.BLUE.value,
    size: Annotated[tuple | str, Body(embed=True)] = PhotoSize.SMALL.value,
):
    # parse input
    # blue white red transform to enum BackgroundColor
    # small large transform to enum PhotoSize
    if isinstance(background_color, str):
        background_color = BackgroundColor[background_color.upper()].value
        # to bgr
        background_color = (
            background_color[2],
            background_color[1],
            background_color[0],
        )
    if isinstance(size, str):
        size = PhotoSize[size.upper()].value

    input_image, extension = parse_base64(image)

    if input_image is None:
        result_message = {"code": 500, "data": None, "message": "failed"}
        return result_message

    res = ID_photo_generator(input_image, background_color, size, extension)

    if res is None:
        result_message = {"code": 500, "data": None, "message": "failed"}
        return result_message

    beauty_image = res[3]
    if beauty_image is None:
        result_message = {"code": 500, "data": None, "message": "failed"}
        return result_message

    # Convert to base64
    res_image = base64.b64encode(beauty_image).decode("utf-8")
    res_image = f"data:image/{extension};base64,{res_image}"

    result_message = {
        "code": 200,
        "data": {
            "image": res_image,
            "height": size[1],
            "width": size[0],
        },
        "message": "success",
    }
    return result_message


if __name__ == "__main__":
    import uvicorn

    # 在8080端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=3030)
