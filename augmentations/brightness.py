import os
import json
import cv2
import numpy as np
import random

def read_labelme_json(json_path):
    """读取 Labelme JSON 文件。"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_image(image_path):
    """从文件中读取图像。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到图像文件: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}. 请检查文件是否损坏或路径是否包含特殊字符。")

    return image

def adjust_brightness(image, brightness_factor):
    """
    调整图像的亮度。

    参数:
        image (np.array): 原始图像。
        brightness_factor (float): 用于调整亮度的因子。

    返回:
        np.array: 亮度调整后的图像。
    """
    # 将图像转换为 float32 以便更精确地调整亮度
    image = image.astype(np.float32)

    # 缩放图像亮度
    image *= brightness_factor

    # 将值裁剪到 [0, 255] 范围内
    image = np.clip(image, 0, 255)

    # 转换回 uint8
    return image.astype(np.uint8)

def update_annotation_for_brightness_change(data, new_image_name):
    """
    更新亮度变化后的图像的标注信息。
    """
    # 更新 imagePath 字段为新的图像名称
    data['imagePath'] = new_image_name

    # 将 imageData 设置为 None
    data['imageData'] = None

    return data

def save_adjusted_image_and_annotations(image, annotations, output_image_path, output_json_path):
    """保存调整后的图像及更新后的标注信息到文件。"""
    cv2.imwrite(output_image_path, image)

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(annotations, file, indent=4, ensure_ascii=False)

def process_dataset(image_dir, bright=True, adjustment_range=(0.2, 0.3)):
    """
    通过调整每个图像的亮度并更新标注信息来处理数据集。

    参数:
        image_dir (str): 包含图像和 JSON 文件的目录。
        bright (bool): 是否增加（变亮）或减少（变暗）亮度。
        adjustment_range (tuple): 亮度调整因子的范围。
    """
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            json_path = image_path.replace('.jpg', '.json').replace('.png', '.json')

            if not os.path.exists(json_path):
                print(f"警告: 未找到图像 {filename} 的 JSON 文件。跳过。")
                continue

            try:
                image = read_image(image_path)
                annotations = read_labelme_json(json_path)

                # 在指定范围内选择随机亮度因子
                if bright:
                    # 使图像变亮
                    brightness_factor = 1 + random.uniform(*adjustment_range)
                    suffix = 'brightened'
                else:
                    # 使图像变暗
                    brightness_factor = 1 - random.uniform(*adjustment_range)
                    suffix = 'darkened'

                # 调整图像亮度
                adjusted_image = adjust_brightness(image, brightness_factor)

                # 生成新的文件名
                base_name, ext = os.path.splitext(filename)
                new_image_name = f"{base_name}_{suffix}{ext}"
                new_json_name = f"{base_name}_{suffix}.json"

                # 用新图像名称更新标注信息
                updated_annotations = update_annotation_for_brightness_change(annotations, new_image_name)

                # 准备输出路径
                output_image_path = os.path.join(image_dir, new_image_name)
                output_json_path = os.path.join(image_dir, new_json_name)

                # 保存结果
                save_adjusted_image_and_annotations(adjusted_image, updated_annotations, output_image_path, output_json_path)
                print(f"已处理 {filename} 并保存 {suffix} 图像和标注信息。")
            except (FileNotFoundError, ValueError) as e:
                print(e)

# 使用该函数来处理您的数据集
image_directory = r'E:\BaiduSyncdisk\CVProject\DLProjects\DataSets\Pepper\IntelPepperV6\t\test'
# 设置 bright=True 以变亮，或 bright=False 以变暗
process_dataset(image_directory, bright=True, adjustment_range=(0.4, 0.5))
