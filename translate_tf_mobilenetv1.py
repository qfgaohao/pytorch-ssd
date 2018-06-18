import torch
import sys

from vision.nn.mobilenet import MobileNetV1
from extract_tf_weights import read_weights


def fill_weights_torch_model(weights, state_dict):
    for name in state_dict:
        if name == 'classifier.weight':
            weight = weights['MobilenetV1/Logits/Conv2d_1c_1x1/weights']
            weight = torch.tensor(weight, dtype=torch.float32).permute(3, 2, 0, 1)
            assert state_dict[name].size() == weight.size()
            state_dict[name] = weight
        elif name == 'classifier.bias':
            bias = weights['MobilenetV1/Logits/Conv2d_1c_1x1/biases']
            bias = torch.tensor(bias, dtype=torch.float32)
            assert state_dict[name].size() == bias.size()
            state_dict[name] = bias
        elif name.endswith('BatchNorm.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/weight', 'BatchNorm/gamma')
            weight = torch.tensor(weights[key], dtype=torch.float32)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        elif name.endswith('BatchNorm.bias'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/bias', 'BatchNorm/beta')
            bias = torch.tensor(weights[key], dtype=torch.float32)
            assert bias.size() == state_dict[name].size()
            state_dict[name] = bias
        elif name.endswith('running_mean'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_mean', 'moving_mean')
            running_mean = torch.tensor(weights[key], dtype=torch.float32)
            assert running_mean.size() == state_dict[name].size()
            state_dict[name] = running_mean
        elif name.endswith('running_var'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_var', 'moving_variance')
            running_var = torch.tensor(weights[key], dtype=torch.float32)
            assert running_var.size() == state_dict[name].size()
            state_dict[name] = running_var
        elif name.endswith('depthwise.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/")
            key = key.replace('depthwise/weight', 'depthwise/depthwise_weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(2, 3, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        else:
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('weight', 'weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(3, 2, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python translate_tf_modelnetv1.py <tf_model.pb> <pytorch_weights.pth>")
    tf_model = sys.argv[1]
    torch_weights_path = sys.argv[2]
    print("Extract weights from tf model.")
    weights = read_weights(tf_model)

    net = MobileNetV1(1001)
    states = net.state_dict()
    print("Translate tf weights.")
    fill_weights_torch_model(weights, states)
    torch.save(states, torch_weights_path)