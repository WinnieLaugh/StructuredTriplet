r"""
Datasets for both the pretraining of the BYOL 
"""

from __future__ import print_function, division
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
import torch
from utils.utils import MyRandomColorJitter, MyRandomResizeCropImgMask, MyRandomRotation
from utils.utils import get_mask, pil_loader, MyRandomResizeCropImgMaskNeg

class BYOLDatasetAug224(Dataset):
    """BYOL dataset"""

    def __init__(self, root_dir, image_size=224):
        """
        Args:
            root_dir (string): root path of the dataset.
            image_size (int): the size of the image to the classification model
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        valid_file_name_file = "{}/code/valid_file_name.txt".format(root_dir)
        with open(valid_file_name_file, "r") as f:
            valid_file_names = f.readlines()

        self.samples = valid_file_names

        self.color_aug = nn.Sequential(
                MyRandomColorJitter(0.8, 1.2, 0.8, 1.2, 0.8, 1.2),
                transforms.RandomApply(
                    torch.nn.ModuleList([transforms.GaussianBlur(3, 1.5)]), p=0.1),
                transforms.Normalize(0.5, 0.5)
            )

        self.trans_aug = nn.Sequential(
                MyRandomResizeCropImgMask(img_size=self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRandomRotation()
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = self.samples[idx]
        file_name = file_name.strip()
        mask_filepath = "{}/mask_type/{}.npy".format(self.root_dir, file_name)

        mask = get_mask(mask_filepath)
        while mask is None:
            file_name = np.random.choice(self.samples)
            file_name = file_name.strip()
            mask_filepath = "{}/mask_type/{}.npy".format(self.root_dir, file_name)
            mask = get_mask(mask_filepath)

        mask = self.transform(mask)
        img_filepath = "{}/imgs/{}.png".format(self.root_dir, file_name)
        img = pil_loader(img_filepath)
        img = self.transform(img)
        img = torch.cat((img, mask), dim=0)
        img_one, img_two = self.trans_aug(img), self.trans_aug(img)

        del img
        del mask
        img_one[:3, :, :] = self.color_aug(img_one[:3, :, :])
        img_two[:3, :, :] = self.color_aug(img_two[:3, :, :])

        return img_one, img_two


class TripletSuperSSDatasetAug224(Dataset):
    """Dataset for our framework"""
    def __init__(self, root_dir, image_size=224, same_prob=0.5):
        """
        Initialization function for the dataset
        Args:
            root_dir (string): root directory of the dataset.
            image_size (int): the size of the image
            same_prob: the probability of the first sampling strategy
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.same_prob = same_prob

        valid_file_name_file = "{}/code/valid_file_name.txt".format(root_dir)
        self.methods = {"TS": 0, "DX": 1, "BS": 0, "MS": 0}
        self.classes = {'acc_tcga': 0, 'blca_tcga': 1, 'brca_tcga': 2,
                        'cesc_tcga': 3, 'chol_tcga': 4, 'coad_tcga': 5,
                        'dlbc_tcga': 6, 'esca_tcga': 7, 'gbm_tcga': 8,
                        'hnsc_tcga': 9, 'kich_tcga': 10, 'kirc_tcga': 11,
                        'kirp_tcga': 12, 'lgg_tcga': 13, 'lihc_tcga': 14,
                        'luad_tcga': 15, 'lusc_tcga': 16, 'meso_tcga': 17,
                        'ov_tcga': 18,   'paad_tcga': 19, 'pcpg_tcga': 20,
                        'prad_tcga': 21, 'read_tcga': 22, 'sarc_tcga': 23,
                        'skcm_tcga': 24, 'stad_tcga': 25, 'tgct_tcga': 26,
                        'thca_tcga': 27, 'thym_tcga': 28, 'ucec_tcga': 29,
                        'ucs_tcga': 30, 'uvm_tcga': 31}

        with open(valid_file_name_file, "r") as f:
            valid_file_names = f.readlines()

        self.samples = valid_file_names

        color_aug = nn.Sequential(
            MyRandomColorJitter(0.8, 1.2, 0.8, 1.2, 0.8, 1.2),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3, 1.5)]), p=0.1),
            transforms.Normalize(0.5, 0.5)
        )

        trans_aug = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            MyRandomRotation()
        )

        crop_instance = nn.Sequential(
            MyRandomResizeCropImgMask(img_size=self.image_size),
        )

        self.color_aug = color_aug
        self.trans_aug = trans_aug
        self.crop_instance = crop_instance
        self.neg_trans_aug = nn.Sequential(
                MyRandomResizeCropImgMaskNeg(img_size=self.image_size)
            )

    def __len__(self):
        return len(self.samples)

    def get_random_img_file(self):
        r"""
        Method to get a random sample in case that the sample retrieved is invalid.
        """
        file_name = np.random.choice(self.samples)
        file_name = file_name.strip()
        mask_filepath = "{}/mask_type/{}.npy".format(self.root_dir, file_name)
        mask = get_mask(mask_filepath)

        return file_name, mask

    def __getitem__(self, idx):
        r"""
        Method to retrieve item from the dataset
        Each item contains a triplet set (anc_img_one, pos_img, neg_img)
                           two attributes of the item (anc_class, anc_method)
                           and an augmented sample for self-supervised learning(anc_img_two)
        """
        file_name = self.samples[idx]
        file_name = file_name.strip()
        mask_filepath = "{}/mask_type/{}.npy".format(self.root_dir, file_name)

        mask = get_mask(mask_filepath)
        while mask is None:
            file_name, mask = self.get_random_img_file()

        mask = self.transform(mask)
        img_filepath = "{}/imgs/{}.png".format(self.root_dir, file_name)
        anc_class_key = file_name.split("/")[0]
        anc_class = self.classes[anc_class_key]
        anc_method_key = file_name.split("-")[5]
        anc_method_key = anc_method_key[:2]
        anc_method = self.methods[anc_method_key]

        img = pil_loader(img_filepath)
        img = self.transform(img)
        img_mask = torch.cat((img, mask), dim=0)
        anc_image = self.crop_instance(img_mask)

        anc_image_one, anc_image_two = self.trans_aug(anc_image), self.trans_aug(anc_image)

        anc_image_one[:3, :, :] = self.color_aug(anc_image_one[:3, :, :])
        anc_image_two[:3, :, :] = self.color_aug(anc_image_two[:3, :, :])

        pos_image = self.crop_instance(img_mask)
        pos_image = self.trans_aug(pos_image)
        pos_image[:3, :, :] = self.color_aug(pos_image[:3, :, :])

        if random.random() < self.same_prob:
            neg_img_mask = img_mask
            neg_image = self.crop_instance(neg_img_mask)
            neg_image = self.neg_trans_aug(neg_image)
            neg_image = self.trans_aug(neg_image)
            neg_image[:3, :, :] = self.color_aug(neg_image[:3, :, :])
        else:
            neg_mask = None
            while neg_mask is None:
                neg_file_name, neg_mask = self.get_random_img_file()

            neg_mask = self.transform(neg_mask)
            neg_img_filepath = "{}/imgs/{}.png".format(self.root_dir, neg_file_name)
            neg_img = pil_loader(neg_img_filepath)
            neg_img = self.transform(neg_img)
            neg_img_mask = torch.cat((neg_img, neg_mask), dim=0)

            neg_image = self.crop_instance(neg_img_mask)
            neg_image = self.trans_aug(neg_image)
            neg_image[:3, :, :] = self.color_aug(neg_image[:3, :, :])
            del neg_img
            del neg_mask

        del img
        del img_mask
        del mask
        del neg_img_mask

        return {'anc_img_one': anc_image_one, 'anc_img_two': anc_image_two,
                    'pos_img': pos_image, 'neg_img': neg_image, 
                    "anc_class": anc_class, "anc_method": anc_method}
