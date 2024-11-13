import cv2
import numpy as np
import gc  # Import garbage collection module
from data_augmentation import augment_image  # Import data augmentation method
from image_normalization import normalize_image  # Import image normalization method


def process_image_for_cnn(img_path, target_size=(64, 64)):
    # 1. Read the image
    img = cv2.imread(img_path)
    
    # Check if the image loaded successfully
    if img is None:
        print(f"Unable to load image: {img_path}")
        return None

    # 2. Image normalization
    img = normalize_image(img)
    
    # Ensure the normalized image is within the [0, 1] range
    img = np.clip(img, 0, 1)

    # 3. Image augmentation
    img = augment_image(img)
    
    # Ensure the augmented image is within the [0, 1] range
    img = np.clip(img, 0, 1)

    # 4. Resize the image to fit the CNN input size
    img = cv2.resize(img, target_size)

    # Release memory no longer needed after processing
    gc.collect()  # Invoke garbage collection to free memory

    # Return the final processed image
    return img


def main():
    # Image path
    img_path = "archive/train/angry/angry1/Training_3908.jpg"  # Adjust image path as needed
    processed_img = process_image_for_cnn(img_path)
    
    if processed_img is not None:
        # Restore the image from [0, 1] range back to [0, 255] range for display
        display_img = (processed_img * 255).astype('uint8')
        cv2.imshow("Processed Image for CNN", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
