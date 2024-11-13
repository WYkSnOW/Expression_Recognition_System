import os
import shutil

def split_images_into_folders(source_folder, destination_folder, images_per_folder):
    # Get all image files
    images = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # Ensure images are sorted in some order (optional)
    images.sort()
    
    # Calculate the number of required folders
    num_folders = (len(images) + images_per_folder - 1) // images_per_folder
    
    for i in range(num_folders):
        # Create a new folder
        new_folder = os.path.join(destination_folder, f'surprise{i+1}')
        os.makedirs(new_folder, exist_ok=True)
        
        # Calculate the range of images for the current folder
        start_idx = i * images_per_folder
        end_idx = min(start_idx + images_per_folder, len(images))
        
        # Move images to the new folder
        for image in images[start_idx:end_idx]:
            src_path = os.path.join(source_folder, image)
            dest_path = os.path.join(new_folder, image)
            shutil.move(src_path, dest_path)
        
        print(f'Created folder {new_folder} with {end_idx - start_idx} images.')

if __name__ == "__main__":
    source_folder = 'archive/test/surprise'  # Folder containing the original images
    destination_folder = 'archive/test/surprise'  # Target path for new folders
    images_per_folder = 1500  # Number of images per folder
    
    split_images_into_folders(source_folder, destination_folder, images_per_folder)
