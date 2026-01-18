'''
图片随机向左或者向右移动，移动距离在moveX与最近边框和边界之间的距离的随机值

向左移动时，最左侧矩形框的左边坐标与图像左侧的距离为基准，移动的距离就是从moveX到这个基准进行随机取值，右侧也是同样道理
'''


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

def calculate_shift_bounds(shapes, image_width):
    """
    根据目标框的坐标计算水平位移范围。

    参数:
        shapes (list): 目标框的形状列表。
        image_width (int): 图像的宽度。

    返回:
        tuple: 最左侧和最右侧目标框的坐标。
    """
    min_left = image_width
    max_right = 0

    for shape in shapes:
        for point in shape['points']:
            if point[0] < min_left:
                min_left = point[0]
            if point[0] > max_right:
                max_right = point[0]

    return min_left, max_right

def shift_image(image, shift):
    """
    对图像进行水平位移，移出的部分进行循环填充。

    参数:
        image (np.array): 原始图像。
        shift (int): 位移量（正值表示右移，负值表示左移）。

    返回:
        np.array: 位移后的图像。
    """
    height, width = image.shape[:2]
    shifted_image = np.zeros_like(image)

    if shift > 0:  # 右移
        shifted_image[:, shift:] = image[:, :width-shift]
        shifted_image[:, :shift] = image[:, width-shift:]
    else:  # 左移
        shifted_image[:, :width+shift] = image[:, -shift:]
        shifted_image[:, width+shift:] = image[:, :-shift]

    return shifted_image

def update_annotation_for_shift(data, shift):
    """
    更新位移后的标注信息。

    参数:
        data (dict): 原始标注数据。
        shift (int): 位移量。

    返回:
        dict: 更新后的标注数据。
    """
    for shape in data['shapes']:
        for point in shape['points']:
            point[0] += shift

    return data

def save_shifted_image_and_annotations(image, annotations, output_image_path, output_json_path):
    """保存位移后的图像及更新后的标注信息到文件。"""
    cv2.imwrite(output_image_path, image)

    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(annotations, file, indent=4, ensure_ascii=False)

def process_dataset(image_dir, moveX=80):
    """
    通过水平位移每个图像并更新标注信息来处理数据集。

    参数:
        image_dir (str): 包含图像和 JSON 文件的目录。
        moveX (int): 决定位移方向的阈值。
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

                # 获取最左侧和最右侧目标框的坐标
                min_left, max_right = calculate_shift_bounds(annotations['shapes'], image.shape[1])

                # 决定位移方向
                image_width = image.shape[1]
                left_space = min_left
                right_space = image_width - max_right

                # 进行位移判断
                if left_space > moveX or right_space > moveX:
                    if left_space > right_space:
                        # 左移，基于左侧边距
                        shift = -random.randint(moveX, int(left_space))
                    else:
                        # 右移，基于右侧边距
                        shift = random.randint(moveX, int(right_space))
                else:
                    print(f"{filename} 的边界距离均小于等于 {moveX}，不进行位移。")
                    continue

                # 进行图像水平位移
                shifted_image = shift_image(image, shift)

                # 更新标注信息
                updated_annotations = update_annotation_for_shift(annotations, shift)

                # 生成新的文件名
                base_name, ext = os.path.splitext(filename)
                new_image_name = f"{base_name}_shifted{ext}"
                new_json_name = f"{base_name}_shifted.json"

                # 更新标注信息中的图像路径
                updated_annotations['imagePath'] = new_image_name
                updated_annotations['imageData'] = None

                # 准备输出路径
                output_image_path = os.path.join(image_dir, new_image_name)
                output_json_path = os.path.join(image_dir, new_json_name)

                # 保存结果
                save_shifted_image_and_annotations(shifted_image, updated_annotations, output_image_path, output_json_path)
                print(f"已处理 {filename} 并保存位移图像和标注信息。")
            except (FileNotFoundError, ValueError) as e:
                print(e)

# 使用该函数来处理您的数据集
image_directory = r'E:\BaiduSyncdisk\CVProject\DLProjects\DataSets\Pepper\IntelPepperV6\t\test'
process_dataset(image_directory, moveX=80)
