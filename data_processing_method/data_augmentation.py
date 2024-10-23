# data_augmentation.py
import cv2
import numpy as np


def augment_image(img):
    # 随机旋转图像
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))

    # 随机平移图像
    tx = np.random.uniform(-10, 10)
    ty = np.random.uniform(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(rotated_img, M, (w, h))

    # 随机水平翻转
    flipped_img = cv2.flip(translated_img, 1) if np.random.random() < 0.5 else translated_img

    return flipped_img


def main():
    # 读取图像文件
    img = cv2.imread("../archive/train/angry/Training_3908.jpg")

    # 增强图像
    augmented_img = augment_image(img)

    # 显示
    # cv2.imshow("Original Image", img)
    cv2.imshow("Augmented Image", augmented_img)

    # 等待键盘输入并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
