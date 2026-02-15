from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import cat
import numpy as np

from model.attention import AttentionNetwork

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Network(nn.Module):
    def __init__(self, layers: list, orthogonal_init: bool = True):
        super().__init__()
        self.net = nn.Sequential(OrderedDict(layers))
        if orthogonal_init:
            self.orthogonal_init()

    def orthogonal_init(self):
        i = 0
        for layer_name, layer in self.net.state_dict().items():
            # The output layer is specially dealt
            gain = 1 if i < len(self.net.state_dict()) - 2 else 0.01
            if layer_name.endswith("weight"):
                nn.init.orthogonal_(layer, gain=gain)
            elif layer_name.endswith("bias"):
                nn.init.constant_(layer, 0)

    def forward(self, x):
        out = self.net(x)
        return out

class MultiObsEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_dim = configs['input_dim']
        output_size = configs['output_size']
        
        # Simple MLP structure similar to DRL
        # DRL: 400 -> 300 -> output
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, output_size),
        )
        
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None
        
        if configs.get('orthogonal_init', True):
            self.orthogonal_init()

    def orthogonal_init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        # Last layer small gain
        if isinstance(self.net[-1], nn.Linear):
             nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs):
        # obs is expected to be a tensor (batch, input_dim)
        out = self.net(obs)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out
