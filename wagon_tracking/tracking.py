from copy import deepcopy

import numpy as np
from sortedcontainers import SortedDict

import vision.utils.box_utils_numpy as box_utils


class WagonTracker:
    def __init__(self, detector, detection_threshold):
        self.detector = detector
        self.elements_info = SortedDict()
        self.wagons_info = SortedDict()
        self.next_drain_id = 0
        self.next_wagon_id = 0
        self.movement_vector = np.array([0.0, 0.0])
        self.detection_threshold = detection_threshold

    def __call__(self, image):
        boxes, labels, _ = self.detector(image)

        self._update_tracking(boxes.numpy(), labels.numpy())
        print(self.wagons_info)
        return deepcopy(self.elements_info)

    def _update_tracking(self, boxes, labels):
        if len(self.elements_info) == 0 and len(boxes) > 0:
            self.next_drain_id = len(boxes)
            self.elements_info = {
                id: (box, lbl) for id, (box, lbl) in enumerate(zip(boxes, labels))
            }
            self._update_wagons(self.elements_info)
            return

        updated_elements_info, new_elements_info = self._update_elements(boxes, labels)

        notfound_elements_info = self._update_notfound_elements(updated_elements_info)

        updated_elements_info.update(notfound_elements_info)

        n_new_boxes = len(new_elements_info)
        if n_new_boxes > 0:
            new_elements_info = {
                self.next_drain_id + id: (box, lbl)
                for id, (box, lbl) in enumerate(new_elements_info)
            }
            updated_elements_info.update(new_elements_info)
            self.next_drain_id += n_new_boxes
            self._update_wagons(new_elements_info)

        self.elements_info = updated_elements_info

    def _update_elements(self, boxes, labels):
        updated_elements_info = {}
        movement_vector = np.array([0.0, 0.0])

        for t_id, (t_box, t_lbl) in self.elements_info.items():
            if len(boxes) == 0:
                break

            ious = box_utils.iou_of(t_box, boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.0:
                u_box = boxes[n_box_idx]

                u_center = (u_box[2:] + u_box[:2]) / 2
                t_center = (t_box[2:] + t_box[:2]) / 2
                movement_vector += u_center - t_center

                updated_elements_info[t_id] = boxes[n_box_idx], labels[n_box_idx]

                boxes = np.delete(boxes, (n_box_idx), axis=0)
                labels = np.delete(labels, (n_box_idx), axis=0)

        if len(updated_elements_info) > 0:
            self.movement_vector = movement_vector / len(updated_elements_info)
        else:
            # This is a hack. At the left end of the frame the object bounding boxes are
            # prone to slightly move their center up. We don't want this o happen.
            self.movement_vector[1] = 0.0

        if len(boxes) > 0:
            new_elements_info = list(zip(boxes, labels))
        else:
            new_elements_info = []

        return updated_elements_info, new_elements_info

    def _update_notfound_elements(self, updated_elements_info: list):
        u_ids = updated_elements_info.keys()

        notfound_elements_info = {}
        for t_id, (t_box, t_lbl) in self.elements_info.items():
            if np.linalg.norm(self.movement_vector) > 5:
                t_box[2:] += self.movement_vector
                t_box[:2] += self.movement_vector

            if t_id not in u_ids:
                notfound_elements_info[t_id] = (t_box, t_lbl)

        return notfound_elements_info

    def _update_wagons(self, new_elements_info):
        for id, (box, _) in new_elements_info.items():
            center = (box[2:] + box[:2]) / 2

            if center[0] <= self.detection_threshold:
                w_info = self.wagons_info.get(self.next_wagon_id - 1, (-1, -1))
                w_info = (w_info[0], id)

                if self.next_wagon_id == 0:
                    # If the drain of the frist wagon was detected on the left side of the
                    # screen, them that wagon is already getting out of the screen, and we
                    # cannot detect the front drain of the second wagon. So the next left
                    # drain that will be detected pertains to the third wagon.
                    self.wagons_info[self.next_wagon_id] = w_info
                    self.next_wagon_id += 2
                else:
                    # If the drain of the first wagon was detected on the left side of the
                    # screen, them the first wagon is still coming, and everything can
                    # happen as expected.
                    self.wagons_info[self.next_wagon_id - 1] = w_info
                    self.next_wagon_id += 1

            else:
                self.wagons_info[self.next_wagon_id] = (id, -1)
