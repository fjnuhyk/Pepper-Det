'''
实现了图像和数据集中的标注框和关键点一起左右翻转

'''

import os
import json
import cv2

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

def flip_image_horizontally(image):
    """Flip the image horizontally."""
    return cv2.flip(image, 1)

def update_annotation_for_flipped_image(data, image_width, new_image_name):
    """
    Update the annotation for a horizontally flipped image.
    This includes updating the coordinates of rectangles and keypoints.
    """
    for shape in data['shapes']:
        # Update each point in the shape
        for point in shape['points']:
            point[0] = image_width - point[0]

    # Update the imagePath to reflect the new image name
    data['imagePath'] = new_image_name

    # Set imageData to None
    data['imageData'] = None

    return data

def save_flipped_image_and_annotations(image, annotations, output_image_path, output_json_path):
    """Save the flipped image and the updated annotations to files."""
    cv2.imwrite(output_image_path, image)

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(annotations, file, indent=4, ensure_ascii=False)

def process_dataset(image_dir):
    """Process the dataset by flipping each image and updating annotations."""
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

                # Flip the image
                flipped_image = flip_image_horizontally(image)

                # Generate new file names
                base_name, ext = os.path.splitext(filename)
                new_image_name = f"{base_name}_flipped{ext}"
                new_json_name = f"{base_name}_flipped.json"

                # Update annotations with the new image name
                updated_annotations = update_annotation_for_flipped_image(annotations, image.shape[1], new_image_name)

                # Prepare output paths
                output_image_path = os.path.join(image_dir, new_image_name)
                output_json_path = os.path.join(image_dir, new_json_name)

                # Save the results
                save_flipped_image_and_annotations(flipped_image, updated_annotations, output_image_path, output_json_path)
                print(f"Processed {filename} and saved flipped image and annotations.")
            except (FileNotFoundError, ValueError) as e:
                print(e)



# Use the function to process your dataset
image_directory = r'E:\BaiduSyncdisk\CVProject\DLProjects\DataSets\Pepper\IntelPepperV6\t'
process_dataset(image_directory)

