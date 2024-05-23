import cv2 as cv
import gradio as gr

from src.beauty import Beauty
from src.enums import BackgroundColor, PhotoSize
from src.matting import Matte
from src.photo_cropper import PhotoCropper


def predict(
    image, background_color=BackgroundColor.BLUE.value, photo_size=PhotoSize.SMALL.value
):
    """
    得到裁剪完成的证件照
    :param image: 输入图像
    :param background_color: 背景颜色
    :param photo_size: 证件照尺寸
    """
    if image is None:
        return ValueError("Image is None")

    # check background color by name
    if not BackgroundColor(background_color):
        raise ValueError("Invalid background color")
    # check photo size by name
    if not PhotoSize(photo_size):
        raise ValueError("Invalid photo size")

    start_time = cv.getTickCount()

    # 人像分割结果和合成图像
    matter = Matte(background_color=background_color)
    matte = matter.matting(image)
    compose_im_with_bg = matter.compose_with_background(image, matte)

    # 获取人脸区域，裁剪证件照
    cropper = PhotoCropper(compose_im_with_bg, matte)
    cropped_img = cropper.crop()

    lined_img = cropper.draw()
    # bgr to rgb
    lined_img = cv.cvtColor(lined_img, cv.COLOR_BGR2RGB)

    # faces detections
    # faces = cropper.copy_image

    # check if the cropped image is None
    if cropped_img is None:
        print("Cropped went wrong. Please try again.")
        return ValueError("Cropped went wrong. Please try again.")

    # 美颜处理
    beauty = Beauty(cropped_img)
    beauty_image = beauty.beautify_skin(5, 8, 8, 0.8)
    beauty_image2 = beauty.sharpen()

    # 修改证件照尺寸。默认为1寸
    # beauty_image = cv.resize(beauty_image, photo_size, interpolation=cv.INTER_AREA)

    # 格式 png jpg

    # 计算时间
    time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    print("time: %.2f s" % time, flush=True)

    return (
        matte,
        compose_im_with_bg,
        cropped_img,
        beauty_image,
        beauty_image2,
        lined_img,
    )


if __name__ == "__main__":
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Image", height=512),
        outputs=[
            gr.Image(type="pil", label="matte", height=512),
            gr.Image(type="pil", label="compose_im_with_bg", height="full"),
            gr.Image(type="pil", label="cropped_img", height="full"),
            gr.Image(type="pil", label="beauty_image", height="full"),
            gr.Image(type="pil", label="beauty_image2", height="full"),
            gr.Image(type="pil", label="lined_img", height="full"),
        ],
        title="ID Photo Generator",
        theme=gr.themes.Base(),
        allow_flagging="never",
    )
    interface.launch(share=False)
