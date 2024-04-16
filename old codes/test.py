import cv2
import numpy as np
from PIL import Image
from IPython.display import display

# Function to create the image mask
def create_mask(image_path, threshold):
    # Load the CXR image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create the mask
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return binary

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

# Function to display the image
def display_image(image):
    display(Image.fromarray(image))

# Function to calculate the average color of an image
def calculate_average_color(image):
    average_color = np.mean(image, axis=(0, 1))
    return average_color[0]

def calculate_threshold(avg_color):
    return avg_color*1.25


# Path to the CXR image
n_image_path = 'train/Normal/06.jpeg'
bp_image_path = 'train/Bacterial Pneumonia/1.jpeg'
# Threshold for creating the mask
# threshold = 185 #n 100 & bp150

# Desired width and height for resizing
desired_width = 300
desired_height = 300


# Overlay with original image
n_image = cv2.imread(n_image_path)
bp_image = cv2.imread(bp_image_path)

n_image = resize_image(n_image, desired_width, desired_height)
bp_image = resize_image(bp_image, desired_width, desired_height)
display_image(n_image)
display_image(bp_image)

# Display average color of the images
average_color1 = calculate_average_color(n_image)
average_color2 = calculate_average_color(bp_image)

threshold1 = calculate_threshold(average_color1)
threshold2 = calculate_threshold(average_color2)
print("Average Color for Normal Image:", average_color1, "\t | threshold:",threshold1 )
print("Average Color for Bacterial Pneumonia Image:",average_color2, "\t | threshold:",threshold2)

# Create the image mask
mask1 = create_mask(n_image_path, threshold1)
mask2 = create_mask(bp_image_path, threshold2)

# Resize the mask
mask1 = resize_image(mask1, desired_width, desired_height)
mask2 = resize_image(mask2, desired_width, desired_height)

# Reverse the color
mask1 = reverse_image(mask1)
mask2 = reverse_image(mask2)

# Set overlay segmentation
scale = 110
x_center = desired_width // 2
y_center = desired_height // 2 - 10
mask1 = apply_radial_selection(mask1, scale, x_center, y_center, view_overlay=False)
mask2 = apply_radial_selection(mask2, scale, x_center, y_center, view_overlay=False)

# Display the resized mask
print("Final Mask for Normal Image:")
display_image(mask1)
print("Final Mask for Bacterial Pneumonia Image:")
display_image(mask2)

# Final result
print("Final Result:")
n_image = overlay_with_mask(n_image, mask1)
bp_image = overlay_with_mask(bp_image, mask2)
display_image(n_image)
display_image(bp_image)
