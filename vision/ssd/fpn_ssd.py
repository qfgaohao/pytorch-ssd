import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

from ..utils import box_utils


class FPNSSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, upsample_mode="nearest"):
        """Compose a SSD model using the given components.
        """
        super(FPNSSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.upsample_mode = upsample_mode

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes if isinstance(t, tuple)])
        self.upsamplers = [
            nn.Upsample(size=(19, 19), mode='bilinear'),
            nn.Upsample(size=(10, 10), mode='bilinear'),
            nn.Upsample(size=(5, 5), mode='bilinear'),
            nn.Upsample(size=(3, 3), mode='bilinear'),
            nn.Upsample(size=(2, 2), mode='bilinear'),
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        features = []
        for end_layer_index in self.source_layer_indexes:

            if isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
            else:
                added_layer = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            start_layer_index = end_layer_index
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            #confidence, location = self.compute_header(header_index, y)
            features.append(y)
            header_index += 1
            # confidences.append(confidence)
            # locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            #confidence, location = self.compute_header(header_index, x)
            features.append(x)
            header_index += 1
            # confidences.append(confidence)
            # locations.append(location)

        upstream_feature = None
        for i in range(len(features) - 1, -1, -1):
            feature = features[i]
            if upstream_feature is not None:
                upstream_feature = self.upsamplers[i](upstream_feature)
                upstream_feature += feature
            else:
                upstream_feature = feature
            confidence, location = self.compute_header(i, upstream_feature)
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=False)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
