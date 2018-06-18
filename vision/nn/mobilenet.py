import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, namedtuple


ConvParam = namedtuple('ConvParam', ['stride', 'depth'])


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, dropout_keep_prob=0.5, depth_multiplier=1.0,
                 min_depth=8, num_feature_layers=0, in_channels=3):
        super(MobileNetV1, self).__init__()

        self.num_classes = num_classes
        self.depth_multiplier = depth_multiplier
        self.dropout_keep_prob = dropout_keep_prob

        # inital normal conv2d, Conv2d_0 layer
        feature_layers = OrderedDict([
            ('Conv2d_0', nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False)),
            ('Conv2d_0/BatchNorm', nn.BatchNorm2d(32, eps=1e-03)),
            ('Conv2d_0/Relu6', nn.ReLU6(inplace=True))
        ])

        # depthwise separable layer params
        conv_params = [
            ConvParam(stride=1, depth=64),
            ConvParam(stride=2, depth=128),
            ConvParam(stride=1, depth=128),
            ConvParam(stride=2, depth=256),
            ConvParam(stride=1, depth=256),
            ConvParam(stride=2, depth=512),
            ConvParam(stride=1, depth=512),
            ConvParam(stride=1, depth=512),
            ConvParam(stride=1, depth=512),
            ConvParam(stride=1, depth=512),
            ConvParam(stride=1, depth=512),
            ConvParam(stride=2, depth=1024),
            ConvParam(stride=1, depth=1024)
        ]

        # depthwise separable Conv2d
        in_channels = 32
        for i, param in enumerate(conv_params):
            if 1 <= num_feature_layers <= i + 1:
                break
            i = i + 1 # make the layer index start from 1 to follow tensorflow MobileNetformat.
            # with groups=output_channels, Conv2d is a depthwise Conv2d.
            feature_layers[f'Conv2d_{i}_depthwise'] = nn.Conv2d(in_channels, in_channels,
                                                                     kernel_size=(3, 3),
                                                                     stride=param.stride,
                                                                     padding=1,
                                                                     groups=in_channels,
                                                                     bias=False)
            feature_layers[f'Conv2d_{i}_depthwise/BatchNorm'] = nn.BatchNorm2d(in_channels, eps=1e-03)
            feature_layers[f'Conv2d_{i}_depthwise/Relu6'] = nn.ReLU6(inplace=True)

            # pointwise Conv2d
            out_channels = max(int(param.depth * depth_multiplier), min_depth)
            feature_layers[f'Conv2d_{i}_pointwise'.format(i)] = nn.Conv2d(in_channels, out_channels,
                                                                     kernel_size=(1, 1), stride=1, bias=False)
            feature_layers[f'Conv2d_{i}_pointwise/BatchNorm'.format(i)] = nn.BatchNorm2d(out_channels, eps=1e-03)
            feature_layers[f'Conv2d_{i}_pointwise/Relu6'.format(i)] = nn.ReLU6(inplace=True)
            in_channels = out_channels

        self.features = nn.Sequential(feature_layers)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        x = self.features(x)
        _, _, height, width = x.size()
        # the kernel size 7x7 is for 224x224 inputs of ImageNet images
        kernel_size = (min(height, 7), min(width, 7))
        x = F.avg_pool2d(x, kernel_size=kernel_size)
        x = F.dropout2d(x, self.dropout_keep_prob)
        x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x
