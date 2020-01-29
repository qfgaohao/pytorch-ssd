import vision.utils.box_utils_numpy as box_utils
import numpy as np

# from collections import namedtuple
from vision.utils.misc import Timer
from copy import deepcopy


# WagonInfo = namedtuple('WagonInfo', 'idx', 'front', 'back')


class WagonTracker:
    def __init__(self, detector):
        self.detector = detector
        self.tracking_info = None
        self.timer = Timer()
        # self.tracked_wagons = []
        # self.wagon_idx = -1

    def __call__(self, image):
        self.timer.start()
        boxes, labels, probs = self.detector(image)
        interval = self.timer.end()

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        self._update_tracking(boxes, labels, probs)
        return deepcopy(self.tracking_info)

    def _update_tracking(self, boxes, labels, probs):
        if len(boxes) == 0:
            return

        if self.tracking_info is None:
            self.tracking_info = list(zip(boxes, labels, probs))
            return

        new_track_info = self._match_boxes(boxes, labels, probs)

        self.tracking_info = new_track_info

    def _match_boxes(self, boxes, labels, probs):
        new_track_info = []

        for t_box, t_lbl, t_prob in self.tracking_info:
            if len(boxes) == 0:
                break

            ious = box_utils.iou_of(t_box, boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.0:
                new_track_info.append(
                    [boxes[n_box_idx], labels[n_box_idx], probs[n_box_idx]]
                )
                boxes = np.delete(boxes, (n_box_idx), axis=0)
                labels = np.delete(labels, (n_box_idx), axis=0)
                probs = np.delete(probs, (n_box_idx), axis=0)
            else:
                new_track_info.append([t_box, t_lbl, t_prob])

        new_track_info.extend(list(zip(boxes, labels, probs)))
        return new_track_info
