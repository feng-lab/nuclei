import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Spatial NL block for image classification.
           [https://github.com/facebookresearch/video-nonlocal-net].
        """

    def __init__(self, in_channels, channels=None, return_att_map=False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels if channels is not None else in_channels // 2
        self.return_att_map = return_att_map
        self.f = nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, bias=False)
        self.h = nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, bias=False)
        self.v = nn.Conv2d(self.channels, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, input_):
        n, _, h, w = input_.size()

        f = self.f(input_).reshape((n, self.channels, h * w))
        g = self.g(input_).reshape((n, self.channels, h * w))
        hh = self.h(input_).reshape((n, self.channels, h * w))

        attention_map = F.softmax(f.transpose(1, 2) @ g, dim=2)
        val = (attention_map @ hh.transpose(1, 2)).transpose(1, 2).reshape((n, self.channels, h, w))

        o = self.v(val)
        attention_feature = o + input_

        if self.return_att_map:
            return attention_feature, attention_map.detach()
        else:
            return attention_feature
