import torch
import numpy as np


def object_detection_collate(batch):
    images = []
    gt_boxes = []
    gt_labels = []
    image_type = type(batch[0][0])
    box_type = type(batch[0][1])
    label_type = type(batch[0][2])
    for image, boxes, labels in batch:
        if image_type is np.ndarray:
            images.append(torch.from_numpy(image))
        elif image_type is torch.Tensor:
            images.append(image)
        else:
            raise TypeError(f"Image should be tensor or np.ndarray, but got {image_type}.")
        if box_type is np.ndarray:
            gt_boxes.append(torch.from_numpy(boxes))
        elif box_type is torch.Tensor:
            gt_boxes.append(boxes)
        else:
            raise TypeError(f"Boxes should be tensor or np.ndarray, but got {box_type}.")
        if label_type is np.ndarray:
            gt_labels.append(torch.from_numpy(labels))
        elif label_type is torch.Tensor:
            gt_labels.append(labels)
        else:
            raise TypeError(f"Labels should be tensor or np.ndarray, but got {label_type}.")
    return torch.stack(images), gt_boxes, gt_labels