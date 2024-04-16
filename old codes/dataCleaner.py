import os
from PIL import Image
import numpy as np


def compare_images(img1, img2):
    return np.array_equal(np.array(img1), np.array(img2))


def process_images(folder):
    with open("debug_dataCleaner.txt", "w") as debug_file:
        for subdir, _, files in os.walk(folder):
            files.sort(key=lambda x: os.path.getsize(os.path.join(subdir, x)))
            for i in range(len(files) - 1):
                img1_path = os.path.join(subdir, files[i])
                img2_path = os.path.join(subdir, files[i + 1])
                print(f"comparing {img1_path} and {img2_path}")
                print(
                    f"comparing {img1_path} and {img2_path}", file=debug_file)
                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)
                if img1.size == img2.size:
                    print(f"!!! Same size: {img1_path} and {img2_path}")
                    print(
                        f"!!! Same size: {img1_path} and {img2_path}", file=debug_file)
                    if compare_images(img1, img2):
                        print("Deleting", img1_path)
                        print("Deleting", img1_path, file=debug_file)
                        print("Compared with", img2_path)
                        print("Compared with", img2_path, file=debug_file)
                        os.remove(img1_path)


process_images("train")
