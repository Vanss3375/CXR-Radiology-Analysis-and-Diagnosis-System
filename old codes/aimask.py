import cv2
import numpy as np
import os

# Functions for image processing (resize_image, reverse_image, apply_radial_selection, overlay_with_mask) remain the same

# Function to resize the image
def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image

# Function to reverse the colors of the image
def reverse_image(image):
    reversed_image = 255 - image
    return reversed_image

# Function to apply radial selection
def apply_radial_selection(image, scale, x_center, y_center, view_overlay=False):
    # Parameters for noise reduction
    opening_size = int(scale * 0.1)  # Adjust as needed for the size of noise to be removed
    closing_size = int(scale * 0.3)  # Adjust as needed for the size of gaps to be filled
    
    # Perform morphological opening to reduce white noise
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size))
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, opening_kernel)
    
    # Perform morphological closing to fill black holes/gaps inside white regions
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, closing_kernel)
    
    # Create a mask for radial selection
    mask = np.zeros_like(image)
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask_area = ((x - x_center) / scale) ** 2 + ((y - y_center) / scale) ** 2 <= 1
    mask[mask_area] = 255
    
    # Apply the mask to the closed image
    result = cv2.bitwise_and(closed_image, closed_image, mask=mask)
    
    if view_overlay:
        # Draw a transparent purple circle overlay
        overlay = closed_image.copy()
        cv2.circle(overlay, (x_center, y_center), int(scale), (128, 0, 128), -1)  # Purple color
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result

# Function to overlay resized original image with the mask created
def overlay_with_mask(original_image, mask):
    # Convert the mask to 3-channel format
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Apply the mask to the original image
    overlaid_image = np.where(mask_rgb == 0, 0, original_image)
    
    return overlaid_image
# Function to create the image mask
def create_mask(image, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create the mask
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Function to save image as PNG
def save_as_png(image, path):
    cv2.imwrite(path, image)

# Function to calculate the average color of an image
def calculate_average_color(image):
    average_color = np.mean(image, axis=(0, 1))
    return average_color[0]

# Function to calculate the threshold based on the average color
def calculate_threshold(avg_color):
    return avg_color * 1.25

# Path to the main folder containing subfolders with images
main_folder = 'train'

# Desired width and height for resizing
desired_width = 300
desired_height = 300

# Set overlay segmentation parameters
scale = 110
x_center = desired_width // 2
y_center = desired_height // 2 - 10

# Function to process images recursively
def process_images(folder):
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, file)
                # Load the image
                image = cv2.imread(image_path)
                # Resize the image
                resized_image = resize_image(image, desired_width, desired_height)
                # Calculate the average color
                avg_color = calculate_average_color(resized_image)
                # Calculate the threshold
                threshold = calculate_threshold(avg_color)
                # Create the mask
                mask = create_mask(resized_image, threshold)
                # Reverse the mask
                reversed_mask = reverse_image(mask)
                # Apply radial selection with noise reduction
                final_mask = apply_radial_selection(reversed_mask, scale, x_center, y_center)
                # Overlay with original image
                final_result = overlay_with_mask(resized_image, final_mask)
                # Determine the category folder
                category_folder = os.path.basename(os.path.dirname(image_path))
                # Save final mask and result
                mask_folder = os.path.join('mask', category_folder)
                fresult_folder = os.path.join('fresult', category_folder)
                os.makedirs(mask_folder, exist_ok=True)
                os.makedirs(fresult_folder, exist_ok=True)
                mask_path = os.path.join(mask_folder, file)
                fresult_path = os.path.join(fresult_folder, file)
                save_as_png(final_mask, mask_path)
                save_as_png(final_result, fresult_path)
                print("Saved:", mask_path, "| and:", fresult_path)

# Process images recursively starting from the main folder
process_images(main_folder)
