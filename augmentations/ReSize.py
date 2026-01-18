'''
数据进行缩小增强，标注框和关键点一起缩小，缩小比例为50%-75%
'''

import os
import json
import cv2
import numpy as np
import random

def read_labelme_json(json_path):
    """Read the Labelme JSON file."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_image(image_path):
    """Read an image from file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image file: {image_path}. Check if the file is corrupted or if the path contains special characters.")

    return image

def resize_and_pad_image(image, scale_factor):
    """
    Resize the image by a scale factor and pad with black to keep the original size.

    Parameters:
        image (np.array): Original image.
        scale_factor (float): Factor by which the image is scaled down.

    Returns:
        np.array: The resized and padded image.
        tuple: The offset (x, y) for placing the scaled image in the center.
    """
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with black padding
    padded_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Compute the offset for centering
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2

    # Place the resized image in the center
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image, x_offset, y_offset, scale_factor

def update_annotation_for_resized_image(data, scale_factor, x_offset, y_offset, new_image_name):
    """
    Update the annotation for a resized image.
    This includes updating the coordinates of rectangles and keypoints.

    Parameters:
        data (dict): Original annotation data.
        scale_factor (float): The scaling factor applied to the image.
        x_offset (int): The x offset for the resized image.
        y_offset (int): The y offset for the resized image.
        new_image_name (str): The new image name to update in annotations.

    Returns:
        dict: Updated annotation data.
    """
    for shape in data['shapes']:
        # Update each point in the shape
        for point in shape['points']:
            point[0] = point[0] * scale_factor + x_offset
            point[1] = point[1] * scale_factor + y_offset

    # Update the imagePath to reflect the new image name
    data['imagePath'] = new_image_name

    # Set imageData to None
    data['imageData'] = None

    return data

def save_resized_image_and_annotations(image, annotations, output_image_path, output_json_path):
    """Save the resized image and the updated annotations to files."""
    cv2.imwrite(output_image_path, image)

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(annotations, file, indent=4, ensure_ascii=False)

def process_dataset(image_dir):
    """
    Process the dataset by resizing each image and updating annotations.

    Parameters:
        image_dir (str): The directory containing the images and JSON files.
    """
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            json_path = image_path.replace('.jpg', '.json').replace('.png', '.json')

            if not os.path.exists(json_path):
                print(f"Warning: No JSON file found for image {filename}. Skipping.")
                continue

            try:
                image = read_image(image_path)
                annotations = read_labelme_json(json_path)

                # Choose a random scale factor between 0.5 and 0.75
                scale_factor = random.uniform(0.5, 0.75)

                # Resize and pad image
                resized_image, x_offset, y_offset, scale_factor = resize_and_pad_image(image, scale_factor)

                # Generate new file names
                base_name, ext = os.path.splitext(filename)
                new_image_name = f"{base_name}_resized{ext}"
                new_json_name = f"{base_name}_resized.json"

                # Update annotations with the new image name
                updated_annotations = update_annotation_for_resized_image(annotations, scale_factor, x_offset, y_offset, new_image_name)

                # Prepare output paths
                output_image_path = os.path.join(image_dir, new_image_name)
                output_json_path = os.path.join(image_dir, new_json_name)

                # Save the results
                save_resized_image_and_annotations(resized_image, updated_annotations, output_image_path, output_json_path)
                print(f"Processed {filename} and saved resized image and annotations.")
            except (FileNotFoundError, ValueError) as e:
                print(e)

# Use the function to process your dataset
image_directory = r'E:\BaiduSyncdisk\CVProject\DLProjects\DataSets\Pepper\IntelPepperV6\t'
process_dataset(image_directory)
