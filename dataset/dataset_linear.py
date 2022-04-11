r"""
Linear dataset for linear evaluation protocal.
"""

from __future__ import print_function, division
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np
from utils.utils import MyRandomResizeCropImgMask, get_mask_linear, pil_loader

dataset_mask = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # NuCLS
                [12, 13, 14, 15, 16, 17, 18], # BreCaHAD
                [19, 20, 21, 22, 23, 24, 25], # CoNSeP
                [26, 27, 28, 29], # MoNuSAC
                [30, 31, 32, 33, 34]] # panNuke


class LinearEvalSingleDataset(Dataset):
    r"""
    Linear Evaluation Dataset.

    Args:
        root_dir: the root folder path of the dataset
        dataset_names: evaluation dataset name list
        dataset_idx: index of the dataset to evaluate on
        split_name: 'train', 'valid', or 'test'
    """
    def __init__(self, root_dir, dataset_idx, split_name="train"):
        self.root_dir = root_dir
        dataset_names = ["CoNSeP", "PanNuke"]
        self.dataset_name = dataset_names[dataset_idx]

        dataset_name = dataset_names[dataset_idx]

        dataset_folder_base = os.path.join(root_dir, dataset_name)
        list_path = os.path.join(dataset_folder_base, "{}.txt".format(split_name))
        npy_path = os.path.join(dataset_folder_base, "{}.npy".format(split_name))

        list_paths = []
        with open(list_path, "r") as f_in:
            file_paths = f_in.readlines()
            for each_line in file_paths:
                each_line = each_line.strip("\n")
                list_paths.append(each_line)

        img_instance_types = []
        img_instance_type_npy = np.load(npy_path)
        for img_instance_type in img_instance_type_npy:
            img_instance_types.append(img_instance_type)

        self.list_paths = list_paths
        self.img_instance_types = img_instance_types

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
        self.crop_instance = nn.Sequential(MyRandomResizeCropImgMask(img_size=224))

    def __len__(self):
        return len(self.img_instance_types)

    def get_list_paths(self):
        r"""
        :return: the list of img paths
        """
        return self.list_paths

    def get_img_instance_types(self):
        r"""
        :return: the list of (dataset_idx, img_index, unique_index, type_this_index) pairs
        """
        return self.img_instance_types

    def __getitem__(self, idx):
        _, img_idx, unique_index, type_this_idx = self.img_instance_types[idx]
        folders = self.list_paths[img_idx].split("/")
        img_path = os.path.join(self.root_dir, self.dataset_name,
                                 "img", folders[0], folders[1])
        img = pil_loader(img_path)
        img = self.transform(img)

        mask_path = os.path.join(self.root_dir, self.dataset_name,
                                 "mask", folders[0], folders[1][:-3] + "npy")

        mask = get_mask_linear(mask_path, unique_index)
        mask = self.transform(mask)

        img_mask = torch.cat((img, mask), dim=0)
        image = self.crop_instance(img_mask)
        image[:3, :, :] = self.color_aug(image[:3, :, :])

        del img
        del mask

        return image, type_this_idx


class LinearEvalSingleDataset20x(Dataset):
    r"""
    Linear Evaluation Dataset.

    Args:
        root_dir: the root folder path of the dataset
        dataset_names: evaluation dataset name list
        dataset_idx: index of the dataset to evaluate on
        split_name: 'train', 'valid', or 'test'
    """
    def __init__(self, root_dir, dataset_idx, split_name="train"):
        self.root_dir = root_dir
        dataset_names = ["CoNSeP", "PanNuke"]
        self.dataset_name = dataset_names[dataset_idx]

        dataset_name = dataset_names[dataset_idx]

        dataset_folder_base = os.path.join(root_dir, dataset_name)
        list_path = os.path.join(dataset_folder_base, "{}.txt".format(split_name))
        npy_path = os.path.join(dataset_folder_base, "{}.npy".format(split_name))

        list_paths = []
        with open(list_path, "r") as f_in:
            file_paths = f_in.readlines()
            for each_line in file_paths:
                each_line = each_line.strip("\n")
                list_paths.append(each_line)

        img_instance_types = []
        img_instance_type_npy = np.load(npy_path)
        for img_instance_type in img_instance_type_npy:
            img_instance_types.append(img_instance_type)

        self.list_paths = list_paths
        self.img_instance_types = img_instance_types

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.color_aug = nn.Sequential(transforms.Normalize(0.5, 0.5))
        self.crop_instance = nn.Sequential(MyRandomResizeCropImgMask(img_size=112))
        self.resize_to_original = nn.Sequential(transforms.Resize(224))

    def __len__(self):
        return len(self.img_instance_types)

    def get_list_paths(self):
        r"""
        :return: the list of img paths
        """
        return self.list_paths

    def get_img_instance_types(self):
        r"""
        :return: the list of (dataset_idx, img_index, unique_index, type_this_index) pairs
        """
        return self.img_instance_types

    def __getitem__(self, idx):
        _, img_idx, unique_index, type_this_idx = self.img_instance_types[idx]
        folders = self.list_paths[img_idx].split("/")
        img_path = os.path.join(self.root_dir, self.dataset_name,
                                 "img", folders[0], folders[1])
        img = pil_loader(img_path)
        img = self.transform(img)

        mask_path = os.path.join(self.root_dir, self.dataset_name,
                                 "mask", folders[0], folders[1][:-3] + "npy")

        mask = get_mask_linear(mask_path, unique_index)
        mask = self.transform(mask)

        img_mask = torch.cat((img, mask), dim=0)
        image = self.crop_instance(img_mask)
        image[:3, :, :] = self.color_aug(image[:3, :, :])
        image = self.resize_to_original(image)

        del img
        del mask

        return image, type_this_idx
