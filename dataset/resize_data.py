import cv2
import os
import random
from cv2 import split
import numpy as np


img_root_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/40x/panNuke/img"
img_dst_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/panNuke/img"
mask_root_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/40x/panNuke/mask"
mask_dst_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/panNuke/mask"
type_root_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/40x/panNuke/type"
type_dst_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/panNuke/type"

label_colors = [[  0,   0,   255], [255, 255,   0],
                [255,   0,   0], [255,   0, 255],
                [  0, 255,   0], [  0, 255, 255],
                [  0,   0, 0], [165, 165, 165],
                [255, 0, 165],   [165, 255, 255],
                [0, 165, 255],   [0, 255, 165]
                ]

def resize():
    for split_name in os.listdir(img_root_dir):
        img_split_folder = os.path.join(img_root_dir, split_name)
        img_dst_folder = os.path.join(img_dst_dir, split_name)
        os.makedirs(img_dst_folder, exist_ok=True)

        mask_split_folder = os.path.join(mask_root_dir, split_name)
        mask_dst_folder = os.path.join(mask_dst_dir, split_name)
        os.makedirs(mask_dst_folder, exist_ok=True)

        type_split_folder = os.path.join(type_root_dir, split_name)
        type_dst_folder = os.path.join(type_dst_dir, split_name)
        os.makedirs(type_dst_folder, exist_ok=True)

        for filename in os.listdir(img_split_folder):
            file_path = os.path.join(img_split_folder, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2), interpolation=cv2.INTER_AREA)

            dst_file_path = os.path.join(img_dst_folder, filename)
            cv2.imwrite(dst_file_path, img)

            file_path = os.path.join(mask_split_folder, filename[:-4]+".npy")
            mask = np.load(file_path)
            mask = cv2.resize(mask, (mask.shape[0]//2, mask.shape[1]//2), interpolation=cv2.INTER_NEAREST)

            dst_filepath = os.path.join(mask_dst_folder, filename[:-4]+".npy")
            with open(dst_filepath, "wb") as f:
                np.save(f, mask)
            
            type_file_path = os.path.join(type_split_folder, filename[:-4] + ".npy")
            type_this = np.load(type_file_path)
            
            type_dst_filepath = os.path.join(type_dst_folder, filename[:-4] + ".npy")
            with open(type_dst_filepath, "wb") as f:
                np.save(f, type_this)


def draw_dilation(img, instance_mask, instance_type=None):
    img_overlay = img.copy()

    if instance_type is not None:
        for instance in instance_type:
            binary_map = np.zeros_like(img, dtype=np.uint8)

            indexes = np.where(instance_mask == instance[0])

            binary_map[indexes] = 255
            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

            img_overlay[inst_pixels_dilated] = label_colors[(instance[1]%len(label_colors))]
            img_overlay[indexes] = img[indexes]
    else:
        for instance_idx in sorted(np.unique(instance_mask))[1:]:
            binary_map = np.zeros_like(img, dtype=np.uint8)
            indexes = np.where(instance_mask == instance_idx)

            binary_map[indexes] = 255
            kernal = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(binary_map, kernal, iterations=1)
            inst_pixels_dilated = np.where((dilation == [255, 255, 255]).all(axis=2))

            img_overlay[inst_pixels_dilated] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            img_overlay[indexes] = img[indexes]

    return img_overlay


def overlay():
    dst_overlay_dir =  "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/panNuke/overlay"
    dst_type_dir = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/panNuke/type"
    for split_name in os.listdir(img_dst_dir):
        img_dst_folder = os.path.join(img_dst_dir, split_name)
        dst_mask_folder = os.path.join(mask_dst_dir, split_name)
        dst_type_folder = os.path.join(dst_type_dir, split_name)
        dst_overlay_folder = os.path.join(dst_overlay_dir, split_name)
        os.makedirs(dst_overlay_folder, exist_ok=True)

        for filename in os.listdir(img_dst_folder):
            img_filepath = os.path.join(img_dst_folder, filename)
            mask_filepath = os.path.join(dst_mask_folder, filename[:-4]+".npy")
            type_filepath = os.path.join(dst_type_folder, filename[:-4]+".npy")
            overlay_filepath = os.path.join(dst_overlay_folder, filename)

            img = cv2.imread(img_filepath)
            mask = np.load(mask_filepath)
            type = np.load(type_filepath)

            overlay = draw_dilation(img, mask, type)
            cv2.imwrite(overlay_filepath, overlay)


def get_sample_list():
    dataset_folder_base = "/mnt/group-ai-medical-2/private/wenhuazhang/data/multi-class/nuclei-level-multi-class/20x/CoNSeP"
    dataset_image_base = os.path.join(dataset_folder_base, "img")

    training_file_list = []
    for split_name in ["Train"]:
        split_folder_path = os.path.join(dataset_image_base, split_name)

        for file_name in os.listdir(split_folder_path):
            training_file_list.append(f"{split_name}/{file_name}\n")
    
    test_file_list = []
    for split_name in ["Test"]:
        split_folder_path = os.path.join(dataset_image_base, split_name)

        for file_name in os.listdir(split_folder_path):
            test_file_list.append(f"{split_name}/{file_name}\n")
    
    img_instance_types = []
    for img_idx, image_path in enumerate(training_file_list):
        mask_path = f"{dataset_folder_base}/mask/{image_path[:-5]}.npy"
        mask = np.load(mask_path)

        type_path = f"{dataset_folder_base}/type/{image_path[:-5]}.npy"
        mask_type = np.load(type_path)

        for inst_id, inst_tp in mask_type:
            inst_indexes = np.where(mask == inst_id)

            if len(inst_indexes[0]) > 5:
                img_instance_types.append([0, img_idx, inst_id, inst_tp])

    img_instance_types_test = []
    for img_idx, image_path in enumerate(test_file_list):
        mask_path = f"{dataset_folder_base}/mask/{image_path[:-5]}.npy"
        mask = np.load(mask_path)

        type_path = f"{dataset_folder_base}/type/{image_path[:-5]}.npy"
        mask_type = np.load(type_path)

        for inst_id, inst_tp in mask_type:
            inst_indexes = np.where(mask == inst_id)

            if len(inst_indexes[0]) > 5:
                img_instance_types_test.append([0, img_idx, inst_id, inst_tp])

    dst_text_path = os.path.join(dataset_folder_base, "train.txt")
    with open(dst_text_path, "w") as f:
        for file_path in training_file_list:
            f.write(file_path)
    
    dst_text_path = os.path.join(dataset_folder_base, "test.txt")
    with open(dst_text_path, "w") as f:
        for file_path in test_file_list:
            f.write(file_path)

    dst_npy_path = os.path.join(dataset_folder_base, "train.npy")
    img_instance_types = np.array(img_instance_types)
    with open(dst_npy_path, "wb") as f:
        np.save(f, img_instance_types)

    dst_npy_path = os.path.join(dataset_folder_base, "test.npy")
    with open(dst_npy_path, "wb") as f:
        np.save(f, img_instance_types_test)


if __name__ == "__main__":
    overlay()