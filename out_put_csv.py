import os
import cv2
import csv
import gc  # 导入垃圾回收模块
from image_processing_pipeline import process_image  # 使用封装的图像处理函数
import re  # 导入正则表达式库，用于去除文件夹名称中的数字

def create_csv_if_not_exists(output_csv):
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # 创建标题：第一个是标签，其余是关键点坐标 (468个关键点，每个2个坐标 x, y)
            header = ['label'] + [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)]
            writer.writerow(header)

def save_keypoints_to_csv(faces, label, output_csv):
    flat_keypoints = [coord for face in faces for point in face for coord in point]
    data_row = [label] + flat_keypoints
    with open(output_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)

def process_dataset(dataset_path, output_csv, batch_size=100, max_images=1500):
    create_csv_if_not_exists(output_csv)
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                label = os.path.basename(root)  # 文件夹名就是label
                label = re.sub(r'\d+$', '', label)  # 使用正则表达式去除label末尾的数字
                img_path = os.path.join(root, file)  # 构建图片的完整路径
                image_paths.append((img_path, label))
                if len(image_paths) >= max_images:  # 达到最大处理数量后停止
                    break
        if len(image_paths) >= max_images:  # 确保在超出最大图片数量后停止继续搜集路径
            break

    # 分批处理图像
    for i in range(0, min(len(image_paths), max_images), batch_size):
        batch = image_paths[i:i + batch_size]
        process_batch(batch, output_csv)
        gc.collect()  # 每处理一批后释放内存

def process_batch(batch, output_csv):
    for img_path, label in batch:
        img, faces = process_image(img_path)  # 处理图像并提取关键点
        if faces:
            save_keypoints_to_csv(faces, label, output_csv)
            print(f"已处理: {img_path}, 面部关键点数量: {len(faces)}")
        else:
            print(f"未检测到面部: {img_path}")

if __name__ == "__main__":
    dataset_path = "archive/train/fear/fear1"
    output_csv = "face_keypoints.csv"
    process_dataset(dataset_path, output_csv, batch_size=5, max_images=1500)  # 每次处理1500张图片
