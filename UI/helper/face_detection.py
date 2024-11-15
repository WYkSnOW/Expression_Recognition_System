# UI/helper/face_detection.py
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, padding=0.2):
    """检测图像中的人脸并返回每个正方形框的位置"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_boxes = []
    for (x, y, w, h) in faces:
        pad = int(padding * max(w, h))
        x, y = max(0, x - pad), max(0, y - pad)
        size = max(w, h) + 2 * pad
        x_end, y_end = min(x + size, frame.shape[1]), min(y + size, frame.shape[0])
        face_boxes.append((x, y, x_end, y_end))
    return face_boxes

def draw_face_boxes(frame, boxes):
    """在图像上绘制给定的方框"""
    for (x, y, x_end, y_end) in boxes:
        cv2.rectangle(frame, (x, y), (x_end, y_end), (255, 0, 0), 2)
