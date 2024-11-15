# UI/helper/utils.py
import cv2
from PIL import Image

def convert_frame_to_image(frame, x, y, x_end, y_end):
    """将cv2帧裁剪并转换为PIL图像"""
    face_region = frame[y:y_end, x:x_end]
    return Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
