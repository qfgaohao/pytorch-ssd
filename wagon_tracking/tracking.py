import vision.utils.box_utils_numpy as box_utils
import numpy as np

# from collections import namedtuple
from vision.utils.misc import Timer
from copy import deepcopy


# WagonInfo = namedtuple('WagonInfo', 'idx', 'front', 'back')


class WagonTracker:
    def __init__(self, detector):
        self.detector = detector
        self.drains_info = {}
        self.timer = Timer()
        self.next_drain_id = 0
        self.movement_vector = np.array([0.0, 0.0])

    def __call__(self, image):
        self.timer.start()
        boxes, labels, probs = self.detector(image)
        interval = self.timer.end()

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        self._update_tracking(boxes.numpy(), labels.numpy())
        return deepcopy(self.drains_info)

    def _update_tracking(self, boxes, labels):
        if len(self.drains_info) == 0 and len(boxes) > 0:
            self.next_drain_id = len(boxes)
            self.drains_info = {
                id: (box, lbl) for id, (box, lbl) in enumerate(zip(boxes, labels))
            }
            return

        updated_drains_info, new_drains_info = self._update_drains(boxes, labels)

        notfound_drains_info = self._update_notfound_drains(updated_drains_info)

        updated_drains_info.update(notfound_drains_info)

        n_new_boxes = len(new_drains_info)
        if n_new_boxes > 0:
            updated_drains_info.update(
                {
                    self.next_drain_id + id: (box, lbl)
                    for id, (box, lbl) in enumerate(new_drains_info)
                }
            )
            self.next_drain_id += n_new_boxes

        self.drains_info = updated_drains_info

    def _update_drains(self, boxes, labels):
        updated_drains_info = {}
        movement_vector = np.array([0.0, 0.0])

        for t_id, (t_box, t_lbl) in self.drains_info.items():
            if len(boxes) == 0:
                break

            ious = box_utils.iou_of(t_box, boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.0:
                u_box = boxes[n_box_idx]

                u_center = (u_box[2:] + u_box[:2]) / 2
                t_center = (t_box[2:] + t_box[:2]) / 2
                movement_vector += u_center - t_center

                updated_drains_info[t_id] = boxes[n_box_idx], labels[n_box_idx]

                boxes = np.delete(boxes, (n_box_idx), axis=0)
                labels = np.delete(labels, (n_box_idx), axis=0)

        if len(updated_drains_info) > 0:
            self.movement_vector = movement_vector / len(updated_drains_info)
        else:
            # This is a hack. At the left end of the frame the object bounding boxes are
            # prone to slightly move their center up. We don't want this o happen.
            self.movement_vector[1] = 0.0

        if len(boxes) > 0:
            new_drains_info = list(zip(boxes, labels))
        else:
            new_drains_info = []

        return updated_drains_info, new_drains_info

    def _update_notfound_drains(self, updated_drains_info: list):
        u_ids = updated_drains_info.keys()

        notfound_drains_info = {}
        for t_id, (t_box, t_lbl) in self.drains_info.items():
            if np.linalg.norm(self.movement_vector) > 5:
                t_box[2:] += self.movement_vector
                t_box[:2] += self.movement_vector

            if t_id not in u_ids:
                notfound_drains_info[t_id] = (t_box, t_lbl)

        return notfound_drains_info
