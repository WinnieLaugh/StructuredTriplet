r"""
Utility functions for datasets and training
"""
from functools import wraps
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class MyRandomRotation(torch.nn.Module):
    """Randomly rotate the tensor by 0, 90, 180, or 270"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        angle = torch.randint(0, 4, size=(1, )).item()
        angle = angle * 90
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
                brightness_factor = torch.tensor(1.0).uniform_(self.brightness_min,
                                                            self.brightness_max).item()
                img = TF.adjust_brightness(img, brightness_factor)

            if fn_id == 1:
                contrast_factor = torch.tensor(1.0).uniform_(self.contrast_min,
                                                            self.contrast_max).item()
                img = TF.adjust_contrast(img, contrast_factor)

            if fn_id == 2:
                saturation_factor = torch.tensor(1.0).uniform_(self.saturation_min,
                                                            self.saturation_max).item()
                img = TF.adjust_saturation(img, saturation_factor)

        return img


class MyRandomResizeCropImgMask(torch.nn.Module):
    """Randomly do the resize then crop with specified parameters"""
    def __init__(self, scale=(0.7, 0.95), img_size=224, interpolation=Image.NEAREST):
        super().__init__()
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.interpolation = interpolation
        self.img_size = img_size

    def forward(self, img):
        width, height = TF.get_image_size(img)
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


def pil_loader(path):
    r"""
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_mask(mask_path):
    r"""
    Function to get an instance mask from LarSPI Dataset
    """
    mask = np.load(mask_path)
    unique_indexes = sorted(np.unique(mask))[1:]

    mask_instance = None

    if len(unique_indexes) > 0:
        unique_index = np.random.choice(unique_indexes)
        pos_mask_indexes = np.where(mask == unique_index)

        try_count = 0
        while ((len(pos_mask_indexes[0]) < 10 and (np.max(pos_mask_indexes[0]) == 1024 or \
                                                    np.max(pos_mask_indexes[1]) == 1024 or \
                                                    np.min(pos_mask_indexes[0]) == 0 or \
                                                    np.min(pos_mask_indexes[1]) == 0))
                                               or np.max(pos_mask_indexes[0]) == np.min(pos_mask_indexes[0])
                                               or np.max(pos_mask_indexes[1]) == np.min(pos_mask_indexes[1])) \
                and try_count < 100:
            unique_index = np.random.choice(unique_indexes)
            pos_mask_indexes = np.where(mask == unique_index)
            try_count += 1

        if try_count < 100:
            mask_instance = np.zeros_like(mask)
            mask_instance[pos_mask_indexes] = 1.

        mask_instance = np.zeros_like(mask)
        mask_instance[pos_mask_indexes] = 1.

    return mask_instance


def get_mask_linear(mask_path, unique_index):
    r"""
    Function to get an instance mask from evaluation datasets
    """
    mask = np.load(mask_path)
    pos_mask_indexes = np.where(mask == unique_index)
    mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    mask[pos_mask_indexes] = 1.

    return mask


def inference(loader, encoder, device):
    r"""
    Function to inference features with the trained encoder
    """
    feature_vector = []
    labels_vector = []

    for x, y in loader:
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
    r"""
    Function to get features and labels for the evaluation dataset
    """
    train_x, train_y = inference(train_loader, encoder, device)
    test_x, test_y = inference(test_loader, encoder, device)
    return train_x, train_y, test_x, test_y


def create_data_loaders_from_arrays(x_train, y_train, x_test, y_test, batch_size):
    r"""
    Function to create dataloaders with the extracted features and labels
    """
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(y_test)
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    r"""
    Utility function to get the device of the module
    """
    return next(module.parameters()).device


def set_requires_grad(model, val):
    r"""
    Fix or release fix of the parameters of the model
    """
    for p in model.parameters():
        p.requires_grad = val


# loss fn
def loss_fn(x, y):
    r"""
    Loss function of the self-supervised branch
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def triplet_loss(anchor, positive, negative, size_average=True, margin=1.):
    r"""
    Loss function of the structured triplet branch
    """
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() if size_average else losses.sum()


class EMA():
    r"""
    Exponetial moving average function
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        r"""
        Return the exponetial moving average of the old and new parameters

        new = old * beta + (1 - beta) * new
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    r"""
    A wrapper function to update the model's parameters by the exponetial moving average function.
    """
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MLP(nn.Module):
    r"""
    MLP class for projector and predictor
    """
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    r"""
    A wrapper class for the base neural network
    will manage the interception of the hidden layer output
    and pipe it into the projecter and predictor nets
    """
    def __init__(self, net, MLP_size, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = MLP(MLP_size, projection_size, projection_hidden_size)
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x, y=None):
        r"""
        Get representation from the model
        The representation if the output feature of the average pooling layer
        """
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'

        if y is not None:
            _ = self.net(y)
            hidden_y = self.hidden
            self.hidden = None
            assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
            return hidden, hidden_y

        return hidden

    def forward(self, x, y=None):
        representation = self.get_representation(x, y)
        projection = self.projector(representation)
        return representation, projection
