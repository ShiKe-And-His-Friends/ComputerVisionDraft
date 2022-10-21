# *********************************************************#
#   预测
#   单张图片  摄像头  FPS  目录遍历
# *********************************************************#
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
