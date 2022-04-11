"""
# code for byol is adapted from https://github.com/lucidrains/byol-pytorch
"""

import copy
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from utils.utils import NetWrapper, EMA, MLP, get_module_device, singleton
from utils.utils import set_requires_grad, update_moving_average, loss_fn, triplet_loss


class Resnet224Model(nn.Module):
    """
    A modification model of the base resnets.
    """
    def __init__(self, input_dim=3, model_name="easy", zero_init=True):
        super().__init__()
        if model_name == "resnet18":
            model = resnet18(zero_init_residual=zero_init)
        elif model_name == "resnet34":
            model = resnet34(zero_init_residual=zero_init)
        elif model_name == "resnet50":
            model = resnet50(zero_init_residual=zero_init)
        elif model_name == "resnet101":
            model = resnet101(zero_init_residual=zero_init)
        elif model_name == "resnet152":
            model = resnet152(zero_init_residual=zero_init)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_dim, self.inplanes, 
                            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        return x


class TripletSupervisedBYOLModel(nn.Module):
    r"""
    Main class of our framework
    Our framework is composed of three branches: Stuctured Triplet branch,
                                                 attribute learning branch,
                                                 and conventional self-supervised branch.
    Details of the three branches can be found in the paper.
    """
    def __init__(
        self,
        net,
        hidden_layer="avgpool",
        projection_size=256, #512,
        projection_hidden_size=4096, #256,
        moving_average_decay=0.99,
        margin=1.0,
        alpha=1.0,
        beta=1.0,
        cancer_types = 32
    ):
        super().__init__()
        mlp_size = 512

        self.online_encoder = NetWrapper(net, mlp_size, projection_size, 
                                        projection_hidden_size, layer=hidden_layer)
        self.super_classifier_disease = MLP(mlp_size, cancer_types, 256)
        self.super_classifier_method = MLP(mlp_size, 2, 256)
        self.triplet_projector = MLP(mlp_size, projection_size, projection_hidden_size)

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
        self.triplet_loss = triplet_loss
        self.supervised_loss = nn.CrossEntropyLoss()

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        r"""
        Reset the target encoder.
        """
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        r"""
        Update the target encoder by exponetial moving average of the old and new parameters (BYOL).
        """
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, anc_img_one, anc_img_two, pos_img, neg_img, disease_gt, method_gt):
        r"""
        Structured Triplet branch: The input triplet is (anc_img_one, pos_img, neg_img)
                                   We get representations of the three inputs and apply triplet loss.
        Attribute Learning branch: The input image is the anc_img_one and the supervision labels
                                    are disease_gt and method_gt.
        Conventional Self-supervised branch: The input couple is (anc_img_one, anc_img_two).
                                    We get representations of the two inputs and apply cosine loss.
        """

        online_representation_one = self.online_encoder.get_representation(anc_img_one)
        online_representation_two = self.online_encoder.get_representation(anc_img_two)

        online_proj_one = self.online_encoder.projector(online_representation_one)
        online_proj_two = self.online_encoder.projector(online_representation_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
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


class BYOL(nn.Module):
    r"""
    Main class of the self-supervised branch.
    Details of this branch can by found in the paper:
        Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond,
        P. H., Buchatskaya, E., ... & Valko, M. (2020).
        Bootstrap your own latent: A new approach to self-supervised learning.
        https://arxiv.org/abs/2006.07733
    """
    def __init__(
        self,
        net,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    ):
        super().__init__()

        mlp_size = 512
        self.online_encoder = NetWrapper(net, mlp_size, projection_size, 
                                        projection_hidden_size, layer=hidden_layer)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        _ = self._get_target_encoder()
        self.to(device)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        r"""
        Reset the target encoder
        """
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        r"""
        Update the target encoder by exponetial moving average of the old and new parameters
        """
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two):
        r"""
        Send the two differently augmented images to the encoders and get their projections
        Get the prediction from one projection and compare it with the other projection, and vice vesa.
        """
        _, online_proj_one = self.online_encoder(image_one)
        _, online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            _, target_proj_one = target_encoder(image_one)
            _, target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

