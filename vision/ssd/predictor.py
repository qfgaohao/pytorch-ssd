import torch.nn.functional as F
import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform


class Predictor:
    def __init__(self, net, size, mean, priors, center_variance, size_variance,
                 iou_threshold, filter_threshold=0.01, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.priors = priors.to(self.device)
        self.net.to(self.device)
        self.net.eval()

    # TODO: SUPPORT MULTIPLE IMAGES
    def predict(self, image, top_k=-1, prob_threshold=None):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            confidences, locations = self.net.forward(images)
        softmax = F.softmax(confidences, dim=2)

        boxes = box_utils.convert_locations_to_boxes(
            locations, self.priors, self.center_variance, self.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes = boxes[0]
        softmax = softmax[0]
        picked_boxes = []
        picked_probs = []
        picked_labels = []
        if not prob_threshold:
            prob_threshold = self.filter_threshold

        for class_index in range(1, softmax.size(1)):
            probs = softmax[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            picked = box_utils.non_maximum_suppression(probs, subset_boxes, self.iou_threshold, top_k)
            probs = probs[picked]
            subset_boxes = subset_boxes[picked, :]
            picked_boxes.append(subset_boxes)
            picked_probs.append(probs)
            picked_labels.extend([class_index] * subset_boxes.size(0))
        if not picked_labels:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_labels = torch.tensor(picked_labels)
        picked_probs = torch.cat(picked_probs)
        picked_boxes = torch.cat(picked_boxes)
        picked_boxes[:, 0] *= width
        picked_boxes[:, 1] *= height
        picked_boxes[:, 2] *= width
        picked_boxes[:, 3] *= height
        return picked_boxes, picked_labels, picked_probs
