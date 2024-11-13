import os
import cv2
import csv
import gc  # Import garbage collection module
from data_processing_method.image_processing_pipeline import process_image

import re  # Import regular expression library for removing numbers from folder names

def create_csv_if_not_exists(output_csv):
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Create headers: the first is the label, the rest are keypoint coordinates (468 keypoints, each with 2 coordinates x, y)
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
                label = os.path.basename(root)  # Folder name is the label
                label = re.sub(r'\d+$', '', label)  # Remove trailing numbers from label using regular expression
                img_path = os.path.join(root, file)  # Construct the full path of the image
                image_paths.append((img_path, label))
                if len(image_paths) >= max_images:  # Stop after reaching the maximum number of images
                    break
        if len(image_paths) >= max_images:  # Ensure to stop collecting paths after exceeding the maximum number of images
            break

    # Process images in batches
    for i in range(0, min(len(image_paths), max_images), batch_size):
        batch = image_paths[i:i + batch_size]
        process_batch(batch, output_csv)
        gc.collect()  # Release memory after processing each batch

def process_batch(batch, output_csv):
    for img_path, label in batch:
        img, faces = process_image(img_path)  # Process image and extract keypoints
        if faces:
            save_keypoints_to_csv(faces, label, output_csv)
            print(f"Processed: {img_path}, Number of face keypoints: {len(faces)}")
        else:
            print(f"No face detected: {img_path}")

if __name__ == "__main__":
    dataset_path = "archive/test/surprise/surprise1"
    output_csv = "face_keypoints_test1.csv"
    process_dataset(dataset_path, output_csv, batch_size=5, max_images=1500)  # Process 1500 images at a time
