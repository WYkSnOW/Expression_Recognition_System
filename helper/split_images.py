import os
import shutil

def split_images_into_folders(source_folder, destination_folder, images_per_folder):
    # 获取所有图片文件
    images = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # 确保图片按照某种顺序排列（可选）
    images.sort()
    
    # 计算所需文件夹的数量
    num_folders = (len(images) + images_per_folder - 1) // images_per_folder
    
    for i in range(num_folders):
        # 创建新的文件夹
        new_folder = os.path.join(destination_folder, f'surprise{i+1}')
        os.makedirs(new_folder, exist_ok=True)
        
        # 计算当前文件夹的图片范围
        start_idx = i * images_per_folder
        end_idx = min(start_idx + images_per_folder, len(images))
        
        # 将图片移动到新的文件夹
        for image in images[start_idx:end_idx]:
            src_path = os.path.join(source_folder, image)
            dest_path = os.path.join(new_folder, image)
            shutil.move(src_path, dest_path)
        
        print(f'Created folder {new_folder} with {end_idx - start_idx} images.')

if __name__ == "__main__":
    source_folder = 'archive/test/surprise'  # 原图片所在文件夹
    destination_folder = 'archive/test/surprise'  # 新文件夹的目标路径
    images_per_folder = 1500  # 每个文件夹包含的图片数量
    
    split_images_into_folders(source_folder, destination_folder, images_per_folder)
