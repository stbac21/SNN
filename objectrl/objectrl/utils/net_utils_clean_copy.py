# -----------------------------------------------------------------------------------
# ObjectRL: An Object-Oriented Reinforcement Learning Codebase
# Copyright (C) 2025 ADIN Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------------

import importlib
from collections.abc import Callable
from typing import Literal

from objectrl.nets.layers.bayesian_layers import (
    BBBLinear,
    CLTLinear,
    CLTLinearDet,
    LRLinear,
)
from objectrl.utils.custom_act import CReLU

import torch
import torch.optim as optim
from torch import nn as nn
import snntorch as snn
from snntorch import surrogate, utils
class SNN(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int=512,
        act: str = "relu",
        has_norm: bool = False,

        T: int = 5, 
        beta: float = 0.95,

        activation_function:any = nn.ReLU(False),
        spike_grad:any = surrogate.fast_sigmoid(slope=2), 

    ) -> None:
        super().__init__()
        width=800 
        self.T=T

        self.model = nn.Sequential(
            nn.Linear(dim_in, width),
            activation_function,
            snn.Leaky(
                beta=beta, 
                threshold=1.0, 
                spike_grad=spike_grad, 
                surrogate_disable=False, 
                init_hidden=True, 
                output=False, 
                inhibition=False, 
                learn_beta=False, 
                learn_threshold=False, 
                reset_mechanism='subtract', 
                reset_delay=True, 
            ),
            nn.Linear(width, width), snn.Leaky(beta=beta, threshold=1.0, spike_grad=spike_grad, surrogate_disable=False, init_hidden=True, output=False, inhibition=False, learn_beta=False, learn_threshold=False, reset_mechanism='subtract', reset_delay=True),
            nn.Linear(width, width), snn.Leaky(beta=beta, threshold=1.0, spike_grad=spike_grad, surrogate_disable=False, init_hidden=True, output=False, inhibition=False, learn_beta=False, learn_threshold=False, reset_mechanism='subtract', reset_delay=True),
            nn.Linear(width, dim_out), snn.Leaky(
                beta=1.0, 
                threshold=1e9, 
                reset_mechanism='none', 
                spike_grad=spike_grad,
                init_hidden=True, 
                inhibition=False, 
                output=True, 
            ),
        )

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        snn.utils.reset(self.model)

        for t in range(self.T):
            _, states = self.model(x)
        return states

class ANN(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        width: int=512,
        act: str = "relu",
        has_norm: bool = False,

    ) -> None:
        super().__init__()
        width=4096
        self.model = nn.Sequential(
            nn.Linear(dim_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), 
            nn.Linear(width, width), nn.ReLU(), 
            nn.Linear(width, width), nn.ReLU(), 
            nn.Linear(width, dim_out), 
        )

    def forward(self, x) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        return self.model(x)

