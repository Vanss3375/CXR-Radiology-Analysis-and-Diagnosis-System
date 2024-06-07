import os
import shutil
import random
from PIL import Image

def copy_random_images(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for root, dirs, files in os.walk(source_folder):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            output_subdir = os.path.join(output_folder, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            images = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= 125:
                selected_images = random.sample(images, 125)
            else:
                selected_images = images
            for i, img_name in enumerate(selected_images):
                img_path = os.path.join(subdir_path, img_name)
                new_img_name = f"{i+1}.{img_name.split('.')[-1]}"
                new_img_path = os.path.join(output_subdir, new_img_name)
                if not img_name.lower().endswith('.jpeg') and not img_name.lower().endswith('.jpg'):
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        new_img_path = os.path.join(output_subdir, f"{i+1}.jpg")
                        img.save(new_img_path, "JPEG")
                else:
                    shutil.copy(img_path, new_img_path)

                print(f"Copying {img_name} to {new_img_path}")

source_folder = "0TestDataset"
output_folder = "1DataSet"
copy_random_images(source_folder, output_folder)
