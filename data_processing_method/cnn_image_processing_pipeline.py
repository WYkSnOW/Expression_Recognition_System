import cv2
import numpy as np
import gc  # 导入垃圾回收模块
from data_augmentation import augment_image  # 导入数据增强方法
from image_normalization import normalize_image  # 导入图像归一化方法


def process_image_for_cnn(img_path, target_size=(64, 64)):
    # 1. 读取图像
    img = cv2.imread(img_path)
    
    # 检查图像是否成功加载
    if img is None:
        print(f"无法加载图像: {img_path}")
        return None

    # 2. 图像归一化
    img = normalize_image(img)
    
    # 确保归一化后的图像在 [0, 1] 范围内
    img = np.clip(img, 0, 1)

    # 3. 图像增强
    img = augment_image(img)
    
    # 确保增强后的图像在 [0, 1] 范围内
    img = np.clip(img, 0, 1)

    # 4. 调整图像大小以适配CNN输入尺寸
    img = cv2.resize(img, target_size)

    # 处理完成后释放不再需要的内存
    gc.collect()  # 调用垃圾回收释放内存

    # 返回最终处理的图像
    return img


def main():
    # 图片路径
    img_path = "archive/train/angry/angry1/Training_3908.jpg"  # 请根据需要调整图片路径
    processed_img = process_image_for_cnn(img_path)
    
    if processed_img is not None:
        # 将图像从 [0, 1] 范围恢复到 [0, 255] 范围以便显示
        display_img = (processed_img * 255).astype('uint8')
        cv2.imshow("Processed Image for CNN", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
