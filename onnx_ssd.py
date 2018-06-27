from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets import voc_dataset

import torch
import sys
import torch.onnx


if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <net type>  <model path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]


num_classes = len(voc_dataset.class_names)
if net_type == "mobilenet-v1-ssd":
    net = create_mobilenetv1_ssd(num_classes)
else:
    net = create_vgg_ssd(num_classes)
net.load(model_path)

dummy_input = torch.randn(1, 3, 300, 300)
torch.onnx.export(net, dummy_input, "mobilenetv1_ssd.proto", verbose=False, output_names=['confidences', 'locations'])
