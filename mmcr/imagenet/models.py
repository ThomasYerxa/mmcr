import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import resnet50
from torch import Tensor
from typing import Tuple
import einops

from composer.models import ComposerModel
from typing import Any, Tuple

import sys

sys.path.append("..")


class MomentumModel(nn.Module):
    def __init__(
        self,
        projector_dims: list = [8192, 8192, 512],
        bias_last=False,
        bias_proj=False,
    ):
        super(MomentumModel, self).__init__()
        # insures output of encoder for all datasets is 2048-dimensional
        self.f = resnet50(zero_init_residual=True)
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [2048] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g = nn.Sequential(*layers)

        # initialize momentum background and projector
        self.mom_f = resnet50(zero_init_residual=True)
        self.mom_f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.mom_g = nn.Sequential(*layers)

        params_f_online, params_f_mom = self.f.parameters(), self.mom_f.parameters()
        params_g_online, params_g_mom = self.g.parameters(), self.mom_g.parameters()

        for po, pm in zip(params_f_online, params_f_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

        for po, pm in zip(params_g_online, params_g_mom):
            pm.data.copy_(po.data)
            pm.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)

        x_momentum = self.mom_f(x)
        feature_momentum = torch.flatten(x_momentum, start_dim=1)
        out_momentum = self.mom_g(feature_momentum)

        return feature, out, feature_momentum, out_momentum


class MomentumComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective):
        super().__init__()

        self.module = module
        self.objective = objective
        self.c = 0  # counts the number of forward calls

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict
        self.c += 1
        return loss

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(batch, Tensor):
            inputs = batch
        else:
            inputs, _ = batch

        features, outputs, features_momentum, outputs_momentum = self.module(inputs)
        if isinstance(batch, Tensor):
            return features, outputs
        else:
            return [outputs, outputs_momentum]

    def get_backbone(self):
        return self.module


class Model(nn.Module):
    def __init__(
        self,
        projector_dims: list = [8192, 8192, 512],
        bias_last=False,
        bias_proj=False,
    ):
        super(Model, self).__init__()

        # insures output of encoder for all datasets is 2048-dimensional
        self.f = resnet50(zero_init_residual=True)
        self.f.fc = nn.Identity()

        # projection head (Following exactly barlow twins offical repo)
        projector_dims = [2048] + projector_dims
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=bias_proj)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=bias_last))
        self.g = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_ = self.f(x)
        feature = torch.flatten(x_, start_dim=1)
        out = self.g(feature)

        return feature, out


class ComposerWrapper(ComposerModel):
    def __init__(self, module: torch.nn.Module, objective):
        super().__init__()

        self.module = module
        self.objective = objective
        self.c = 0

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        loss, loss_dict = self.objective(outputs)
        self.loss_dict = loss_dict
        self.c += 1
        return loss

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(batch, Tensor):
            inputs = batch
        else:
            inputs, _ = batch

        features, outputs = self.module(inputs)
        if isinstance(batch, Tensor):
            return features, outputs
        else:
            return outputs

    def get_backbone(self):
        return self.module
