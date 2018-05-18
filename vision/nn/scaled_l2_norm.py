import torch.nn as nn
import torch
import torch.nn.functional as F


class ScaledL2Norm(nn.Module):
    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return (F.normalize(x, p=2, dim=1)
                * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3))

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)