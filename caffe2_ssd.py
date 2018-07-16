from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets import voc_dataset

import torch
import sys
import torch.onnx
from caffe2.python.onnx.backend import Caffe2Backend as c2
import onnx
from caffe2.python.predictor import mobile_exporter


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
net.eval()

model_path = "models/mobilenetv1_ssd.onnx"
init_net_path = "models/mobilenetv1_ssd_init_net.pb"
init_net_txt_path = "models/mobilenetv1_ssd_init_net.pbtxt"
predict_net_path = "models/mobilenetv1_ssd_predict_net.pb"
predict_net_txt_path = "models/mobilenetv1_ssd_predict_net.pbtxt"

dummy_input = torch.randn(1, 3, 300, 300)
torch.onnx.export(net, dummy_input, model_path, verbose=True, output_names=['scores', 'boxes'])

model = onnx.load(model_path)
#prepared_backend = onnx_caffe2.backend.prepare(model)

# extract the workspace and the model proto from the internal representation
# c2_workspace = prepared_backend.workspace
# c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = c2.onnx_graph_to_caffe2_net(model)

# Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
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