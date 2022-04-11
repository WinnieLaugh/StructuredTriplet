import copy
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

# helper functions
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
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def triplet_loss(anchor, positive, negative, size_average=True, margin=1.):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() if size_average else losses.sum()


def triplet_normalize_loss(anchor, positive, negative, size_average=True, margin=1.):
    anchor = F.normalize(anchor, dim=-1, p=2)
    positive = F.normalize(positive, dim=-1, p=2)
    negative = F.normalize(negative, dim=-1, p=2)

    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() if size_average else losses.sum()


def triplet_cos_loss(anchor, positive, negative, size_average=True, margin=1.):
    anchor = F.normalize(anchor, dim=-1, p=2)
    positive = F.normalize(positive, dim=-1, p=2)
    negative = F.normalize(negative, dim=-1, p=2)

    distance_positive = 2 - 2 * (anchor * positive).sum(dim=-1)
    distance_negative = 2 - 2 * (anchor * negative).sum(dim=-1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() if size_average else losses.sum()

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size))

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = MLP(512, projection_size, projection_hidden_size)
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


class NetWrapper2(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = MLP(512*4, projection_size, projection_hidden_size)
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

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projection = self.projector(representation)
        return representation, projection

class BYOL_Triplet_Supervised_Model(nn.Module):
    def __init__(
        self,
        net,
        hidden_layer="avgpool",
        projection_size=256, #512,
        projection_hidden_size=4096, #256,
        moving_average_decay=0.99,
        use_momentum=True,
        margin=1.0,
        alpha=1.0,
        beta=1.0,
        cancer_types = 32,
        resnet_simple=True,
        triplet_loss_fn = "ori"
    ):
        super().__init__()
        if resnet_simple:
            self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
            self.super_classifier_disease = MLP(512, cancer_types, 256)
            self.super_classifier_method = MLP(512, 2, 256)
            self.triplet_projector = MLP(512, projection_size, projection_hidden_size)
        else:
            self.online_encoder = NetWrapper2(net, projection_size, projection_hidden_size, layer=hidden_layer)
            self.super_classifier_disease = MLP(512*4, cancer_types, 256)
            self.super_classifier_method = MLP(512*4, 2, 256)
            self.triplet_projector = MLP(512*4, projection_size, projection_hidden_size)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)
        self.margin = margin
        self.loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.target_encoder = self._get_target_encoder()
        if triplet_loss_fn == "ori":
            self.triplet_loss = triplet_loss
        elif triplet_loss_fn == "normalize":
            self.triplet_loss = triplet_normalize_loss
        elif triplet_loss_fn == "cos":
            self.triplet_loss = triplet_cos_loss

        self.supervised_loss = nn.CrossEntropyLoss()

        print("alpha, ", self.alpha)
        print("beta, ", self.beta)

    def set_byol_fixed(self, fix_byol_parameters=True):
        if fix_byol_parameters:
            for params in self.online_encoder.net.parameters():
                params.requires_grad = False
        else:
            for params in self.online_encoder.net.parameters():
                params.requires_grad = True

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, anc_img_one, anc_img_two, pos_img, neg_img, disease_gt, method_gt):
        online_representation_one = self.online_encoder.get_representation(anc_img_one)
        online_representation_two = self.online_encoder.get_representation(anc_img_two)

        online_proj_one = self.online_encoder.projector(online_representation_one)
        online_proj_two = self.online_encoder.projector(online_representation_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            _, target_proj_one = target_encoder(anc_img_one)
            _, target_proj_two = target_encoder(anc_img_two)

        # byol loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        # supervised loss
        disease_kind = self.super_classifier_disease(online_representation_one)
        super_disease_loss = self.supervised_loss(disease_kind, disease_gt)

        method_kind = self.super_classifier_method(online_representation_one)
        super_method_loss = self.supervised_loss(method_kind, method_gt)

        # triplet loss
        representation_two = self.online_encoder.get_representation(pos_img)
        representation_three = self.online_encoder.get_representation(neg_img)
        triplet_representation_one = self.triplet_projector(online_representation_one)
        triplet_representation_two = self.triplet_projector(representation_two)
        triplet_representation_three = self.triplet_projector(representation_three)

        loss_triplet = self.triplet_loss(triplet_representation_one, triplet_representation_two,
                                         triplet_representation_three, margin=self.margin)

        loss_byol = (loss_one + loss_two).mean()
        loss_super = super_disease_loss.mean() + super_method_loss.mean()
        loss = loss_byol + self.alpha *loss_super + self.beta*loss_triplet

        return loss, loss_byol, loss_super, loss_triplet

    def tune_linear(self, anc_img_one, anc_img_two, pos_img, neg_img, disease_gt, method_gt):
        with torch.no_grad():
            online_representation_one = self.online_encoder.get_representation(anc_img_one).detach()
            online_representation_two = self.online_encoder.get_representation(anc_img_two).detach()

        online_proj_one = self.online_encoder.projector(online_representation_one)
        online_proj_two = self.online_encoder.projector(online_representation_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            _, target_proj_one = target_encoder(anc_img_one)
            _, target_proj_two = target_encoder(anc_img_two)

        # byol loss
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        # supervised loss
        disease_kind = self.super_classifier_disease(online_representation_one)
        super_disease_loss = self.supervised_loss(disease_kind, disease_gt)

        method_kind = self.super_classifier_method(online_representation_one)
        super_method_loss = self.supervised_loss(method_kind, method_gt)

        # triplet loss
        representation_two = self.online_encoder.get_representation(pos_img)
        representation_three = self.online_encoder.get_representation(neg_img)
        triplet_representation_one = self.triplet_projector(online_representation_one)
        triplet_representation_two = self.triplet_projector(representation_two)
        triplet_representation_three = self.triplet_projector(representation_three)

        loss_triplet = self.triplet_loss(triplet_representation_one, triplet_representation_two,
                                         triplet_representation_three, margin=self.margin)

        loss_byol = (loss_one + loss_two).mean()
        loss_super = super_disease_loss.mean() + super_method_loss.mean()
        loss = loss_byol + self.alpha*loss_super + self.beta*loss_triplet

        return loss, loss_byol, loss_super, loss_triplet