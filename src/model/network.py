import torch
import torch.nn as nn
import numpy as np

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
