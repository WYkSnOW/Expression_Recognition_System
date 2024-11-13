import cv2
import numpy as np
import gc  # Import garbage collection module
from data_augmentation import augment_image  # Import data augmentation method
from image_normalization import normalize_image  # Import image normalization method
from face_mesh_module import FaceMeshDetector  # Import face keypoint detection class


def process_image(img_path):
    # 1. Read the image
    img = cv2.imread(img_path)

    # Check if the image loaded successfully
    if img is None:
        print(f"Unable to load image: {img_path}")
        return None, None

    # 2. Image normalization
    img = normalize_image(img)

    # Ensure the normalized image is within the [0, 1] range
    img = np.clip(img, 0, 1)

    # 3. Image augmentation
    img = augment_image(img)

    # Ensure the augmented image is within the [0, 1] range
    img = np.clip(img, 0, 1)

    # 4. Face keypoint extraction
    detector = FaceMeshDetector()
    img = cv2.resize(img, (256, 256))  # Resize for MediaPipe processing
    img = (img * 255).astype('uint8')
    final_img, faces = detector.find_face_mesh(img, draw=True)

    # Release memory no longer needed after processing
    del img
    gc.collect()  # Invoke garbage collection to free memory

    # Return the final processed image
    return final_img, faces


def main():
    # Image path
    img_path = "archive/train/angry/angry1/Training_3908.jpg"  # Adjust image path as needed
    img, faces = process_image(img_path)
    print(faces)
    if img is not None:
        cv2.imshow("Processed Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
