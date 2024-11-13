# data_augmentation.py
import cv2
import numpy as np


def augment_image(img):
    # Randomly rotate the image
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (w, h))

    # Randomly translate the image
    tx = np.random.uniform(-10, 10)
    ty = np.random.uniform(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(rotated_img, M, (w, h))

    # Random horizontal flip
    flipped_img = cv2.flip(translated_img, 1) if np.random.random() < 0.5 else translated_img

    return flipped_img


def main():
    # Read the image file
    img = cv2.imread("../archive/train/angry/Training_3908.jpg")

    # Augment the image
    augmented_img = augment_image(img)

    # Display
    # cv2.imshow("Original Image", img)
    cv2.imshow("Augmented Image", augmented_img)

    # Wait for keyboard input and close window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
