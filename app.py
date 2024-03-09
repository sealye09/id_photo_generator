import cv2 as cv
import gradio as gr

from src.beauty import Beauty
from src.enums import BackgroundColor, MattingModel, PhotoSize
from src.matting import inference
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
        return ValueError("Cropped went wrong. Please try again.")

    # 美颜处理
    beauty = Beauty(cropped_img)
    beauty_image = beauty.beautify_skin(5, 12, 12, 1.2)

    # 修改证件照尺寸。默认为1寸
    cropped_img = cv.resize(cropped_img, photo_size, interpolation=cv.INTER_AREA)

    # 格式 png jpg

    # 计算时间
    time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    print("time: %.2f s" % time, flush=True)

    return matte, compose_im_with_bg, cropped_img, beauty_image


if __name__ == "__main__":
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Image", height=512),
        outputs=[
            gr.Image(type="pil", label="Matte", height=512),
            gr.Image(type="pil", label="Compose Image", height="full"),
            gr.Image(type="pil", label="Face Area", height="full"),
            gr.Image(type="pil", label="Beauty Image", height="full"),
        ],
        title="person-modnet",
        theme=gr.themes.Base(),
        allow_flagging="never",
    )
    interface.launch(share=False)
