# image_normalization.py
import cv2


def normalize_image(img):
    normalized_img = img.astype('float32') / 255.0
    return normalized_img


def main():
    # Read the image file
    img = cv2.imread("../archive/train/angry/Training_3908.jpg")

    # Normalize the image
    normalized_img = normalize_image(img)

    # Display
    # cv2.imshow("Original Image", img)
    cv2.imshow("Normalized Image", normalized_img)

    # Wait for keyboard input and close window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
