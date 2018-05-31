import torch
import torch.nn as nn
import logging

from ..utils.model_book import ModelBook


class ModelPrunner:
    def __init__(self, model, train_fun):
        """ Implement the pruning algorithm described in the paper https://arxiv.org/pdf/1611.06440.pdf .

        The prunning criteria is dC/dh * h, while C is the cost, h is the activation.
        """
        self.model = model
        self.train_fun = train_fun
        self.book = ModelBook(self.model)
        self.outputs = {}
        self.grads = {}
        self.handles = []
        self.batch_norms = {}
        self.last_conv = None
        self.convs = {}
        self.linears = {}

    def _make_new_conv(self, conv, filter_index, channel_type="out"):
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"The module is not Conv2d, but {type(conv)}.")

        if channel_type == "out":
            new_conv = nn.Conv2d(conv.in_channels, conv.out_channels - 1, conv.kernel_size, conv.stride,
                                 conv.padding, conv.dilation, conv.groups, conv.bias is not None)
            if filter_index == 0:
                new_conv.weight.data = conv.weight.data[filter_index + 1:]
                if conv.bias is not None:
                    new_conv.bias.data = conv.bias.data[filter_index + 1:]
            elif filter_index == conv.weight.data.size(0) - 1:
                new_conv.weight.data = conv.weight.data[:filter_index]
                if conv.bias is not None:
                    new_conv.bias.data = conv.bias.data[:filter_index]
            else:
                new_conv.weight.data = torch.cat([
                    conv.weight.data[:filter_index],
                    conv.weight.data[filter_index + 1:]
                ])
                if conv.bias is not None:
                    new_conv.bias.data = torch.cat([
                        conv.bias.data[:filter_index],
                        conv.bias.data[filter_index + 1:]
                    ])
        elif channel_type == 'in':
            new_conv = nn.Conv2d(conv.in_channels - 1, conv.out_channels, conv.kernel_size, conv.stride,
                                 conv.padding, conv.dilation, conv.groups, conv.bias is not None)
            if filter_index == 0:
                new_conv.weight.data = conv.weight.data[:, filter_index + 1:, :, :]
            elif filter_index == conv.weight.data.size(1) - 1:
                new_conv.weight.data = conv.weight.data[:, :filter_index, :, :]
            else:
                new_conv.weight.data = torch.cat([
                    conv.weight.data[:, :filter_index, :, :],
                    conv.weight.data[:, filter_index + 1:, :, :]
                ], dim=1)
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data
        else:
            raise ValueError(f"{channel_type} should be either 'in' or 'out'.")
        return new_conv

    def remove_filter(self, conv, filter_index):
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(f"The module is not Conv2d, but {type(conv)}.")

        new_conv = self._make_new_conv(conv, filter_index, channel_type="out")
        path = self.book.get_path(conv)
        logging.info(f"Prune filter {filter_index} on Conv2d {conv} at path {'/'.join(path)}")
        conv_parent = self.book.get_module(path[:-1])
        conv_parent._modules[path[-1]] = new_conv
        self.book.update(path, new_conv)

        next_conv = self.convs.get(conv)
        if next_conv is not None:
            new_next_conv = self._make_new_conv(next_conv, filter_index, channel_type="in")
            next_path = self.book.get_path(next_conv)
            next_parent = self.book.get_module(next_path[:-1])
            next_parent._modules[next_path[-1]] = new_next_conv
            self.book.update(next_path, new_next_conv)

        # reduce the num_features of batch norm
        batch_norm = self.batch_norms.get(conv)
        if batch_norm:
            new_batch_norm = nn.BatchNorm2d(batch_norm.num_features - 1)
            batch_norm_path = self.book.get_path(batch_norm)
            batch_norm_parent = self.book.get_module(batch_norm_path[:-1])
            batch_norm_parent._modules[batch_norm_path[-1]] = new_batch_norm
            self.book.update(batch_norm_path, new_batch_norm)

        # reduce the in channels of linear layer
        linear = self.linears.get(conv)
        if linear:
            new_linear = self._make_new_linear(conv, linear, filter_index)
            linear_path = self.book.get_path(linear)
            linear_parent = self.book.get_module(linear_path[:-1])
            linear_parent._modules[linear_path[-1]] = new_linear
            self.book.update(linear_path, new_linear)

    def _make_new_linear(self, conv, linear, filter_index):
        block = int(linear.in_features / conv.out_channels)
        new_linear = nn.Linear(linear.in_features - block, linear.out_features,
                               bias=linear.bias is not None)
        if filter_index == 0:
            new_linear.weight.data = linear.weight.data[:, block:]
        elif filter_index == conv.out_channels - 1:
            new_linear.weight.data = linear.weight.data[:, :-block]
        else:
            start_index = filter_index * block
            end_index = (filter_index + 1) * block
            new_linear.weight.data = torch.cat([
                linear.weight.data[:, :start_index],
                linear.weight.data[:, end_index:]
            ], dim=1)
        if linear.bias is not None:
            new_linear.bias.data = linear.bias.data
        return new_linear

    def prune(self, ignored_paths=[]):
        """Prune one conv2d filter.
        """
        self.register_hooks(ignored_paths)
        self.train_fun(self.model)
        ranks = []
        for m, output in self.outputs.items():
            output = output.data
            grad = self.grads[m].data
            v = grad * output
            v = v.sum(0).sum(1).sum(1)  # sum to the channel axis.
            v = torch.abs(v)
            v = v / torch.sqrt(torch.sum(v * v))  # normalize
            for i, e in enumerate(v):
                ranks.append((m, i, e))
        module, filter_index, value = min(ranks, key=lambda t: t[2])
        self.remove_filter(module, filter_index)
        self.deregister_hooks()

    def register_hooks(self, ignored_paths=[]):
        """Run register before training for pruning."""
        ignored_modules = set(self.book.get_module(path) for path in ignored_paths)
        self.outputs.clear()
        self.grads.clear()
        self.handles.clear()
        self.last_conv = None
        self.batch_norms.clear()
        self.convs.clear()
        self.linears.clear()

        def forward_hook(m, input, output):
            if isinstance(m, nn.Conv2d):
                if m not in ignored_modules:
                    self.outputs[m] = output
                if self.last_conv:
                    self.convs[self.last_conv] = m
                self.last_conv = m
            elif isinstance(m, nn.BatchNorm2d):
                if self.last_conv:
                    self.batch_norms[self.last_conv] = m
            elif isinstance(m, nn.Linear):
                if self.last_conv:
                    self.linears[self.last_conv] = m
                self.last_conv = None  # after a linear layer the conv layer doesn't matter

        def backward_hook(m, input, output):
            self.grads[m] = output[0]

        for path, m in self.book.modules(module_type=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            h = m.register_forward_hook(forward_hook)
            self.handles.append(h)
            h = m.register_backward_hook(backward_hook)
            self.handles.append(h)

    def deregister_hooks(self):
        """Run degresiter before retraining to recover the model"""
        for handle in self.handles:
            handle.remove()