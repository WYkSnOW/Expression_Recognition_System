import cv2
import numpy as np
import gc  # 导入垃圾回收模块
from data_processing_method.data_augmentation import augment_image  # 导入数据增强方法
from data_processing_method.image_normalization import normalize_image  # 导入图像归一化方法
from data_processing_method.face_mesh_module import FaceMeshDetector  # 导入面部关键点检测类


def process_image(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)

    # 检查图像是否成功加载
    if img is None:
        print(f"无法加载图像: {img_path}")
        return None, None

    # 2. 图像归一化
    img = normalize_image(img)

    # 确保归一化后的图像在 [0, 1] 范围内
    img = np.clip(img, 0, 1)

    # 3. 图像增强
    img = augment_image(img)

    # 确保增强后的图像在 [0, 1] 范围内
    img = np.clip(img, 0, 1)

    # 4. 面部关键点提取
    detector = FaceMeshDetector()
    img = cv2.resize(img, (256, 256))  # 调整大小以便MediaPipe处理
    img = (img * 255).astype('uint8')
    final_img, faces = detector.find_face_mesh(img, draw=True)

    # 处理完成后释放不再需要的内存
    del img
    gc.collect()  # 调用垃圾回收释放内存

    # 返回最终处理的图像
    return final_img, faces


def main():
    # 图片路径
    img_path = "archive/train/angry/Training_3908.jpg"  # 请根据需要调整图片路径
    img, faces = process_image(img_path)
    print(faces)
    if img is not None:
        cv2.imshow("Processed Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
