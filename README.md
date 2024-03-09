# ID Photo Generator

## 项目简介

一个自动生成证件照的工具。使用人脸检测和抠图技术，自动抠出人脸并生成证件照。

-   matting use [MODNet](https://github.com/ZHKKKe/MODNet)
-   face detection use [OpenCV](https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detect.py)

## 安装依赖

```bash
# python 3.10+
# pip 22.3.0+
pip install -r requirements.txt
```

## 使用

```bash
# web ui
python app.py
# api
python api.py
```
