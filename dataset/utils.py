import torchvision.transforms.functional as TF
import random
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import torchvision
import numpy as np
import cv2
import random

Large_Margin = 1000000
label_colors = [[  0,   0,   255], [255, 255,   0],
                [255,   0,   0], [255,   0, 255],
                [  0, 255,   0], [  0, 255, 255],
                [  0,   0, 0], [165, 165, 165],
                [255, 0, 165],   [165, 255, 255],
                [0, 165, 255],   [0, 255, 165]
                ]


def data_debug(data_sample1, iter_count):
    images_anc = data_sample1 / 2 + 0.5

    for count in range(images_anc.shape[0]):
        os.makedirs("test/{}/".format(iter_count), exist_ok=True)

        anc_image = torchvision.transforms.functional.to_pil_image(images_anc[count][:3, :, :])
        anc_image.save("test/{}/{}_anc.png".format(iter_count, count))
        if images_anc[count].shape[0] > 3:
            anc_image = torchvision.transforms.functional.to_pil_image(images_anc[count][3:, :, :])
            anc_image.save("test/{}/{}_anc_mask.png".format(iter_count, count))

    print("saved data debug")


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


def inference(loader, encoder, device):
    feature_vector = []
    labels_vector = []

    for step, (x, y) in enumerate(loader):
        x = x.to(device).float()

        # get encoding
        with torch.no_grad():
            h = encoder.get_representation(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(encoder, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, encoder, device)
    test_X, test_y = inference(test_loader, encoder, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


class BatchPool(nn.Module):
    def forward(self, input):
        if input.shape[0] > 1:
            return torch.mean(input, 0)
        else:
            return input[0]


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# cosine loss fn
def cosine_distance(x, y, temperature=1):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return cos(x, y) / temperature


def get_triplet_loss(x, x_pos, x_neg, debug=False):

    pos_dist = cosine_distance(x, x_pos.detach())
    neg_dist = cosine_distance(x, x_neg.detach())

    pos_exp = torch.exp(pos_dist)  # exp(z_i * z_p / tau)
    neg_exp = torch.exp(neg_dist)

    loss = (-1) * torch.log(pos_exp / (pos_exp + neg_exp))

    if debug:
        return loss, pos_exp, neg_exp
    else:
        return loss


def get_regulation_loss(x, x_regulation):
    loss = 0
    useful_num = 0
    for x_index, regulation_instance in enumerate(x_regulation):
        if regulation_instance is not None:
            x_index_now = x[x_index:x_index+1].repeat(regulation_instance.shape[0], 1)
            loss_now = (2 - 2 * cosine_distance(x_index_now, regulation_instance)).mean()
            loss += loss_now
            useful_num += 1

    return loss / useful_num


def crop_single_instance(img, centroid, target_length=64):
    h, w = img.shape[1:]

    sx, sy, ex, ey = [centroid[1]-target_length//2, centroid[0]-target_length//2,
                      centroid[1]+target_length//2, centroid[0]+target_length//2]

    dh, dw = ey - sy, ex - sx
    res = torch.zeros((3, dh, dw))

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[:, dsy:dey, dsx:dex] = img[:, sy:ey, sx:ex]
    return res


# loss fn
def distengle_loss(global_proj_one, global_proj_two, instance_proj_one, instance_proj_two, env_proj_one, env_proj_two):
    length = global_proj_one.shape[1]
    global_instance_proj_one, global_env_proj_one = torch.split(global_proj_one.detach(), [length//2, length//2], dim=1)
    global_instance_proj_two, global_env_proj_two = torch.split(global_proj_two, [length//2, length//2], dim=1)
    instance_proj_two, env_proj_two = instance_proj_two.detach(), env_proj_two.detach()

    global_instance_proj_one = F.normalize(global_instance_proj_one, dim=-1, p=2)
    instance_proj_one = F.normalize(instance_proj_one, dim=-1, p=2)
    global_env_proj_one = F.normalize(global_env_proj_one, dim=-1, p=2)
    env_proj_one = F.normalize(env_proj_one, dim=-1, p=2)

    global_instance_proj_two = F.normalize(global_instance_proj_two, dim=-1, p=2)
    instance_proj_two = F.normalize(instance_proj_two, dim=-1, p=2)
    global_env_proj_two = F.normalize(global_env_proj_two, dim=-1, p=2)
    env_proj_two = F.normalize(env_proj_two, dim=-1, p=2)

    return 2 - 2 * (global_instance_proj_one * instance_proj_one).sum(dim=-1), \
            2 - 2 * (global_env_proj_one * env_proj_one).sum(dim=-1), \
           2 - 2 * (global_instance_proj_two * instance_proj_two).sum(dim=-1), \
           2 - 2 * (global_env_proj_two * env_proj_two).sum(dim=-1)


def distengle_loss_single(global_feature, instance_feature, env_feature, global_detach=False):
    if global_detach:
        global_feature = global_feature.detach()
    else:
        instance_feature = instance_feature.detach()
        env_feature = env_feature.detach()

    length = global_feature.shape[1]
    global_instance_pred, global_env_pred = torch.split(global_feature, [length//2, length//2], dim=1)

    global_instance_pred = F.normalize(global_instance_pred, dim=-1, p=2)
    global_env_pred = F.normalize(global_env_pred, dim=-1, p=2)

    instance_feature = F.normalize(instance_feature, dim=-1, p=2)
    env_feature = F.normalize(env_feature, dim=-1, p=2)

    return (2 - 2 * (global_instance_pred * instance_feature).sum(dim=-1) +
            2 - 2 * (global_env_pred * env_feature).sum(dim=-1)) / 2


def crop_img(img, mask, target_width=64, image_size=128, instance_idx=None):
    instances_list = []
    instances_types_list = []
    instance_idx_in_list = None

    for channel_index in range(mask.shape[0]-1):
        instance_ids = torch.unique(mask[channel_index, :, :], sorted=True)

        if len(instance_ids) > 1:
            for instance_id in instance_ids[1:]:
                indexes = torch.nonzero((mask[channel_index, :, :] == instance_id), as_tuple=False)

                if instance_idx is not None and instance_id == instance_idx and instance_idx_in_list is None:
                    instance_idx_in_list = len(instances_list)

                image_to_crop = torch.zeros_like(img)
                image_to_crop[:, indexes[:, 0], indexes[:, 1]] = img[:, indexes[:, 0], indexes[:, 1]]

                x_min, x_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
                y_min, y_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

                centroid = [(x_max + x_min) // 2, (y_max + y_min) // 2]

                if x_min == 0 or y_min == 0 or x_max == (image_size-1) or y_max == (image_size-1):
                    if indexes.shape[0] < 200:
                        continue

                if (x_max - x_min < target_width) and (y_max - y_min < target_width):
                    instance_now = crop_single_instance(image_to_crop, centroid, target_length=target_width)
                    if instance_now.shape[1] != 0 and instance_now.shape[2] != 0:
                        instances_list.append(instance_now)
                        instances_types_list.append(channel_index)
                else:
                    max_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
                    instance_now = crop_single_instance(image_to_crop, centroid, target_length=max_length)
                    instance_now = TF.resize(instance_now, (target_width, target_width), interpolation=PIL.Image.NEAREST)
                    if instance_now.shape[1] != 0 and instance_now.shape[2] != 0:
                        instances_list.append(instance_now)
                        instances_types_list.append(channel_index)

    if instance_idx is None:
        return instances_list, instances_types_list
    else:
        return instances_list, instances_types_list, instance_idx_in_list


def crop_single_instance(img, centroid, target_length=64):
    """
    :param img: original img
    :param centroid: centroid of the instance
    :param target_length: target size of the cropped instance
    :return: cropped instance
    """
    height, width = img.shape[1:]

    start_x, start_y, end_x, end_y = [centroid[1]-target_length//2, centroid[0]-target_length//2,
                      centroid[1]+target_length//2, centroid[0]+target_length//2]

    delta_height, delta_width = end_y - start_y, end_x - start_x
    res = torch.zeros((img.shape[0], delta_height, delta_width))
    padding_bool = [False, False, False, False]

    if start_x < 0:
        start_x, dsx = 0, -start_x
        # padding left
        padding_bool[0] = True
    else:
        dsx = 0

    if end_x > width:
        end_x, dex = width, delta_width - (end_x - width)
        # padding right
        padding_bool[1] = True
    else:
        dex = delta_width

    if start_y < 0:
        start_y, dsy = 0, -start_y
        # padding up
        padding_bool[2] = True
    else:
        dsy = 0

    if end_y > height:
        end_y, dey = height, delta_height - (end_y - height)
        # padding down
        padding_bool[3] = True
    else:
        dey = delta_height

    res[:, dsy:dey, dsx:dex] = img[:, start_y:end_y, start_x:end_x]

    if padding_bool[0]:
        res[:, :, :dsx] = (res[:, :, dsx:dsx * 2]).flip(2)

    if padding_bool[1]:
        res[:, :, dex:] = (res[:, :, dex - (delta_width - dex):dex]).flip(2)

    if padding_bool[2]:
        res[:, :dsy, :] = (res[:, dsy:dsy * 2:, :]).flip(1)

    if padding_bool[3]:
        res[:, dey:, :] = (res[:, dey - (delta_height - dey):dey, :]).flip(1)

    return res


def crop_img_instance_with_bg(img, mask, target_width=64):
    """
    :param img: original img
    :param mask: mask of the instance
    :param target_width: target size of the cropped patch
    :return: cropped patch
    """
    indexes = torch.nonzero((mask > 0.99), as_tuple=False)

    x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()
    y_min, y_max = torch.min(indexes[:, 2]).item(), torch.max(indexes[:, 2]).item()

    centroid = [(x_max + x_min) // 2, (y_max + y_min) // 2]

    if (x_max - x_min < target_width) and (y_max - y_min < target_width):
        instance_now = crop_single_instance(img, centroid, target_length=target_width)
        return instance_now

    max_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
    instance_now = crop_single_instance(img, centroid, target_length=max_length)
    instance_now = TF.resize(instance_now, (target_width, target_width), \
                             interpolation=PIL.Image.NEAREST)
    return instance_now


def crop_img_instance_distance(img, mask, target_width=64, image_size=256):
    instances_list = []
    instances_types_list = []
    centroid_list = []

    for channel_index in range(mask.shape[0]-1):
        instance_ids = torch.unique(mask[channel_index, :, :], sorted=True)

        if len(instance_ids) > 1:
            for instance_id in instance_ids[1:]:
                indexes = torch.nonzero((mask[channel_index, :, :] == instance_id), as_tuple=False)

                image_to_crop = torch.zeros_like(img)
                image_to_crop[:, indexes[:, 0], indexes[:, 1]] = img[:, indexes[:, 0], indexes[:, 1]]

                x_min, x_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
                y_min, y_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

                centroid = [(x_max + x_min) // 2, (y_max + y_min) // 2]

                if (x_max - x_min < target_width) and (y_max - y_min < target_width):
                    instance_now = crop_single_instance(image_to_crop, centroid, target_length=target_width)
                    if instance_now.shape[1] != 0 and instance_now.shape[2] != 0:
                        instances_list.append(instance_now)
                        instances_types_list.append(channel_index)
                        centroid_list.append(centroid)
                else:
                    max_length = x_max - x_min if (x_max - x_min) > (y_max - y_min) else y_max - y_min
                    instance_now = crop_single_instance(image_to_crop, centroid, target_length=max_length)
                    instance_now = TF.resize(instance_now, (target_width, target_width), interpolation=PIL.Image.NEAREST)
                    if instance_now.shape[1] != 0 and instance_now.shape[2] != 0:
                        instances_list.append(instance_now)
                        instances_types_list.append(channel_index)
                        centroid_list.append(centroid)

    return instances_list, instances_types_list, centroid_list


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MyRandomRotation(torch.nn.Module):
    """Randomly rotate the tensor by 0, 90, 180, or 270"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        angle = torch.randint(0, 4, size=(1, )).item()
        angle = angle * 90
        return TF.rotate(x, angle)


class MyRandomResizeCrop:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyRandomColorJitter(torch.nn.Module):
    """Randomly do the color jitter with specified range"""
    def __init__(self, brightness_min=0.7, brightness_max=1.3, contrast_min=0.7, contrast_max=2.0,
                 saturation_min=0.7, saturation_max=2.0):
        super().__init__()
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max

    def forward(self, img):
        fn_idx = torch.randperm(3)
        for fn_id in fn_idx:
            if fn_id == 0:
                brightness_factor = torch.tensor(1.0).uniform_(self.brightness_min, self.brightness_max).item()
                img = TF.adjust_brightness(img, brightness_factor)

            if fn_id == 1:
                contrast_factor = torch.tensor(1.0).uniform_(self.contrast_min, self.contrast_max).item()
                img = TF.adjust_contrast(img, contrast_factor)

            if fn_id == 2:
                saturation_factor = torch.tensor(1.0).uniform_(self.saturation_min, self.saturation_max).item()
                img = TF.adjust_saturation(img, saturation_factor)

        return img


class MyRandomResizeCrop(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 1.0), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        scale_factor = torch.tensor(1.0).uniform_(self.scale_min, self.scale_max).item()
        width, height = TF._get_image_size(img)

        w = int(round(width * scale_factor))
        h = int(round(height * scale_factor))

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return TF.resized_crop(img, i, j, h, w, self.img_size, self.interpolation)


class MyFixedResizeCropImgMask(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 0.95), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        width, height = TF._get_image_size(img)
        w = 224
        h = 224

        indexes = torch.nonzero((img[-1, :, :] > 0.9), as_tuple=False)

        y_min, y_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
        x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

        if (x_max - x_min) > w or (y_max - y_min) > h:
            new_length = max(x_max - x_min, y_max - y_min)
            w = new_length
            h = new_length

        x_right = min(x_min, width - w) + 1
        y_right = min(y_min, height - h) + 1

        x_left = max(0, x_max - w)
        y_left = max(0, y_max - h)

        i = (y_left + y_right) // 2
        j = (x_left + x_right) // 2

        return TF.resized_crop(img, i, j, h, w, self.img_size, self.interpolation)


class MyRandomResizeCropImgMask(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 0.95), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        width, height = TF._get_image_size(img)
        w = 224
        h = 224

        indexes = torch.nonzero((img[-1, :, :] > 0.9), as_tuple=False)

        y_min, y_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
        x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

        if (x_max - x_min) > w or (y_max - y_min) > h:
            new_length = max(x_max - x_min, y_max - y_min)
            w = new_length
            h = new_length

        x_right = min(x_min, width - w) + 1
        y_right = min(y_min, height - h) + 1

        x_left = max(0, x_max - w)
        y_left = max(0, y_max - h)

        i = torch.randint(y_left, y_right, size=(1,)).item()
        j = torch.randint(x_left, x_right, size=(1,)).item()

        return TF.resized_crop(img, i, j, h, w, self.img_size, self.interpolation)


class MyRandomResizeCropImgMaskNeg(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 1.0), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        width, height = TF._get_image_size(img)

        indexes = torch.nonzero((img[-1, :, :] > 0.9), as_tuple=False)

        y_min, y_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
        x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

        center = [(y_min + y_max)//2, (x_min + x_max)//2]

        y_left = (y_max - y_min) // 2 + 1
        y_right = height - (y_max - y_min) // 2 - 1
        x_left = (x_max - x_min)//2 + 1
        x_right = width - (x_max - x_min) // 2 - 1

        if y_left >= y_right or x_left >= x_right:
            img_img = img[:-1, :, :]
            img_img = TF.rotate(img_img, 90)
            img[:-1, :, :] = img_img
        else:
            rand_i = torch.randint(y_left, y_right, size=(1,)).item()
            rand_j = torch.randint(x_left, x_right, size=(1,)).item()

            img[-1, :, :] = 0.
            img[-1, indexes[:, 0] - center[0] + rand_i, indexes[:, 1] - center[1] + rand_j] = 1.

        return img


class MyRandomResizeCropImgMaskNegDebug(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 1.0), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        width, height = TF._get_image_size(img)

        indexes = torch.nonzero((img[3, :, :] > 0.9), as_tuple=False)

        y_min, y_max = torch.min(indexes[:, 0]).item(), torch.max(indexes[:, 0]).item()
        x_min, x_max = torch.min(indexes[:, 1]).item(), torch.max(indexes[:, 1]).item()

        center = [(y_min + y_max)//2, (x_min + x_max)//2]

        y_left = (y_max - y_min) // 2 + 1
        y_right = height - (y_max - y_min) // 2 - 1
        x_left = (x_max - x_min)//2 + 1
        x_right = width - (x_max -x_min) // 2 - 1

        if y_left >= y_right or x_left >= x_right:
            img_img = img[:3, :, :]
            img_img = TF.rotate(img_img, 90)
            img[:3, :, :] = img_img
        else:
            rand_i = torch.randint(y_left, y_right, size=(1,)).item()
            rand_j = torch.randint(x_left, x_right, size=(1,)).item()

            img[3, :, :] = 0.
            img[3, indexes[:, 0] - center[0] + rand_i, indexes[:, 1] - center[1] + rand_j] = 1.

        return img


class MyRandomWhiteMask(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self):
        super().__init__()

    def forward(self, img):
        indexes = torch.nonzero((img[-1, :, :] > 0.9), as_tuple=False)

        img_copy = torch.zeros_like(img)
        img_copy[:, indexes[:, 0], indexes[:, 1]] = img[:, indexes[:, 0], indexes[:, 1]]

        return img_copy