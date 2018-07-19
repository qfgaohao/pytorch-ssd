from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets import voc_dataset

import sys
import torch.onnx
from caffe2.python.onnx.backend import Caffe2Backend as c2
import onnx


if len(sys.argv) < 3:
    print('Usage: python convert_to_caffe2_models.py <net type: mobilenet-v1-ssd|others>  <model path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]


num_classes = len(voc_dataset.class_names)
if net_type == "mobilenet-v1-ssd":
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
else:
    net = create_vgg_ssd(num_classes, is_test=True)
net.load(model_path)
net.eval()

model_path = f"models/{net_type}.onnx"
init_net_path = f"models/{net_type}_init_net.pb"
init_net_txt_path = f"models/{net_type}_init_net.pbtxt"
predict_net_path = f"models/{net_type}_predict_net.pb"
predict_net_txt_path = f"models/{net_type}_predict_net.pbtxt"

dummy_input = torch.randn(1, 3, 300, 300)
torch.onnx.export(net, dummy_input, model_path, verbose=False, output_names=['scores', 'boxes'])

model = onnx.load(model_path)
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model)

print(f"Save the model in binary format to the files {init_net_path} and {predict_net_path}.")

with open(init_net_path, "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open(predict_net_path, "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

print(f"Save the model in txt format to the files {init_net_txt_path} and {predict_net_txt_path}. ")
with open(init_net_txt_path, 'w') as f:
    f.write(str(init_net))

with open(predict_net_txt_path, 'w') as f:
    f.write(str(predict_net))