import torch
from torch import nn as nn
import torch.nn.functional as F
from .utils import scale_points_with_weights

class BaselineSampler(nn.Module):
    def __init__(self, origin_channels = 3, direction_channels = 3, output_channels = 40, far = 6, near = 2):
        super(BaselineSampler, self).__init__()
        self.w1 = 256
        self.w2 = 128
        self.origin_channels = origin_channels
        self.direction_channels = direction_channels
        self.output_channels = output_channels
        self.far = far
        self.near = near

        self.layers = nn.ModuleList(
            [
                nn.Linear(self.origin_channels+self.direction_channels, self.w1),
                nn.Linear(self.w1,self.w2),
                nn.Linear(self.w2, self.w2),
            ]
        )

        self.last = nn.Linear(self.w2, self.output_channels)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):

        inputs = torch.cat([rays_o, rays_d], -1)

        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
            outputs = F.relu(outputs)

        outputs = self.last(outputs)
        outputs = F.sigmoid(outputs)

        z_vals = self.near * (1 - outputs) + self.far * outputs
        z_vals, _ = z_vals.sort(dim=-1)

        return scale_points_with_weights(z_vals, rays_o, rays_d), z_vals