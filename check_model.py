import os
import cv2
import joblib  # 用于加载模型
import numpy as np
from image_processing_pipeline import process_image  # 用于图像处理和面部关键点提取

# 加载随机森林模型
def load_rf_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"模型加载成功：{model_path}")
        return model
    except Exception as e:
        print(f"加载模型失败：{e}")
        return None

# 使用模型预测表情
def predict_expression(model, faces):
    if not faces:
        print("未检测到面部关键点")
        return None

    # 将面部关键点扁平化为一维数组，模型的输入应匹配训练时的格式
    flat_keypoints = np.array([coord for face in faces for point in face for coord in point]).reshape(1, -1)

    # 使用模型预测表情
    prediction = model.predict(flat_keypoints)
    return prediction

# 主函数，处理图片并识别表情
def recognize_expression(image_path, model_path):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return

    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图片: {image_path}")
        return

    # 调用图像处理管道，提取面部关键点
    _, faces = process_image(image_path)

    # 加载随机森林模型
    model = load_rf_model(model_path)
    if model is None:
        return

    # 预测表情
    expression = predict_expression(model, faces)
    if expression is not None:
        print(f"预测的表情: {expression[0]}")  # 打印预测结果
    else:
        print("未能预测表情")

if __name__ == "__main__":
    # 图片路径和模型路径
    image_path = "archive/train/sad/Training_2913.jpg"  # 使用绝对路径
    model_path = "rf_model.pkl"  # 已训练好的随机森林模型路径

    # 调用识别函数
    recognize_expression(image_path, model_path)
