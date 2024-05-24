import os
import json
from glob import glob
from PIL import Image

def yolo_to_labelme(image_dir, yolo_dir, output_dir, class_names):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a list of all YOLO annotation files
    yolo_files = glob(os.path.join(yolo_dir, '*.txt'))

    all_shapes = []
    
    for yolo_file in yolo_files:
        with open(yolo_file, 'r') as file:
            yolo_data = file.readlines()
        
        # Construct the base filename without extension
        base_filename = os.path.splitext(os.path.basename(yolo_file))[0]
        
        # Find the corresponding image file with supported extensions
        supported_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None
        for ext in supported_extensions:
            potential_image_path = os.path.join(image_dir, base_filename + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break
        
        if image_path is None:
            print(f"Image file for {yolo_file} not found. Skipping...")
            continue
        
        # Get image dimensions
        with Image.open(image_path) as img:
            image_height = img.height
            image_width = img.width

        # Process each line in the YOLO annotation file
        for line in yolo_data:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert YOLO format to Labelme format
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            shape = {
                "label": class_names[int(class_id)],
                "points": [
                    [x_min, y_min],
                    [x_max, y_max]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            
            shape['imagePath'] = os.path.basename(image_path)
            shape['imageHeight'] = image_height
            shape['imageWidth'] = image_width
            
            all_shapes.append(shape)
    
    # Create the final Labelme JSON structure
    labelme_data = {
        "version": "4.5.6",  # The version of Labelme, adjust as necessary
        "flags": {},
        "shapes": all_shapes,
        "imagePath": None,  # Can be left as None since each shape contains imagePath
        "imageData": None,  # Can be left as None
        "imageHeight": None,  # Can be left as None
        "imageWidth": None    # Can be left as None
    }
    
    # Output JSON filename
    json_filename = 'compiled_annotations.json'
    json_path = os.path.join(output_dir, json_filename)
    
    # Save the Labelme JSON file
    with open(json_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)

# Example usage:
image_directory = './img'
yolo_txt_directory = './bbox-yolo'
output_directory = './yolo'
class_names_list = ['left_lobe', 'right_lobe', 'heart', 'diaphragm', 'abnormalities']  # Replace with your actual class names

yolo_to_labelme(image_directory, yolo_txt_directory, output_directory, class_names_list)
