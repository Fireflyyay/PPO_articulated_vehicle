from typing import List, OrderedDict

import torch
import torch.nn as nn
from torch import cat

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
        embed_size = configs['embed_size']
        hidden_size = configs['hidden_size']
        activate_func = [nn.LeakyReLU(), nn.Tanh()][configs['use_tanh_activate']]
        self.use_img = False if configs['img_shape'] is None else True
        self.use_action_mask = False if configs['action_mask_shape'] is None else True
        self.use_attention = False if configs['attention_configs'] is None else True
        self.input_action = 'input_action_dim' in configs and configs['input_action_dim'] > 0

        if not self.use_attention:
            if configs['n_hidden_layers'] == 1:
                layers = [nn.Linear(configs['n_modal']*embed_size, configs['output_size'])]
            else:
                layers = [nn.Linear(configs['n_modal']*embed_size, hidden_size)]
                for _ in range(configs['n_hidden_layers']-2):
                    layers.append(activate_func)
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Linear(hidden_size, configs['output_size']))
            self.net = nn.Sequential(*layers)
        else:
            attention_configs = configs['attention_configs']
            self.net = AttentionNetwork(
                embed_size,
                attention_configs['depth'],
                attention_configs['heads'],
                attention_configs['dim_head'],
                attention_configs['mlp_dim'],
                configs['n_modal'],
                attention_configs['hidden_dim'],
                configs['output_size'],
            )
        self.output_layer = nn.Tanh() if configs['use_tanh_output'] else None

        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)

        if configs['target_shape'] is not None:
            layers = [nn.Linear(configs['target_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_tgt = nn.Sequential(*layers)
            
        if configs['action_mask_shape'] is not None:
            layers = [nn.Linear(configs['action_mask_shape'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_am = nn.Sequential(*layers)

        # Removed ImgEncoder usage

        if self.input_action:
            layers = [nn.Linear(configs['input_action_dim'], embed_size)]
            for _ in range(configs['n_embed_layers']-1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_action = nn.Sequential(*layers)

    def forward(self, obs):
        embeddings = []
        if hasattr(self, 'embed_lidar'):
            embeddings.append(self.embed_lidar(obs['lidar']))
        if hasattr(self, 'embed_tgt'):
            embeddings.append(self.embed_tgt(obs['target']))
        if hasattr(self, 'embed_am'):
            embeddings.append(self.embed_am(obs['action_mask']))
        if hasattr(self, 'embed_action'):
            embeddings.append(self.embed_action(obs['action']))
        
        x = torch.stack(embeddings, dim=1)
        out = self.net(x)
        if self.output_layer is not None:
            out = self.output_layer(out)
        return out
