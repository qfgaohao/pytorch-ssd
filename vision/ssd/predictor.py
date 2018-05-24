import torch.nn.functional as F
import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform


class Predictor:
    def __init__(self, net, size, mean, priors, center_variance, size_variance, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.priors = priors.to(self.device)
        self.net.to(self.device)
        self.net.eval()

    def predict0(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
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
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        softmax = softmax.to(cpu_device)

        for class_index in range(1, softmax.size(1)):
            probs = softmax[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            picked = box_utils.non_maximum_suppression(probs, subset_boxes, self.iou_threshold, top_k, self.candidate_size)
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


    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
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
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        softmax = softmax.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, softmax.size(1)):
            probs = softmax[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]