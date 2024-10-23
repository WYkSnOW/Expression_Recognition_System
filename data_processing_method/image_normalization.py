# image_normalization.py
import cv2


def normalize_image(img):
    normalized_img = img.astype('float32') / 255.0
    return normalized_img


def main():
    # 读取图像文件
    img = cv2.imread("../archive/train/angry/Training_3908.jpg")

    # 归一化图像
    normalized_img = normalize_image(img)

    # 显示
    # cv2.imshow("Original Image", img)
    cv2.imshow("Normalized Image", normalized_img)

    # 等待键盘输入并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
