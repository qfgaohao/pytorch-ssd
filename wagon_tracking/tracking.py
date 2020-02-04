from copy import deepcopy

import cv2 as cv
import numpy as np
from sortedcontainers import SortedDict

import vision.utils.box_utils_numpy as box_utils

from wagon_tracking.transforms import ImageDownscaleTransform


class MovementEstimator:
    def __init__(self, update_interval=5, frame_downscale_factor=None):
        self.update_interval = update_interval
        self.frame_count = 0

        self.downscale_t = None
        if frame_downscale_factor:
            if (
                not isinstance(frame_downscale_factor, int)
                or frame_downscale_factor < 0
            ):
                raise TypeError('Downscale factor must be an positive integer.')
            self.downscale_t = ImageDownscaleTransform(frame_downscale_factor)

        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.old_gray = None
        self.corners = None

    def __call__(self, frame):
        if self.downscale_t:
            frame = self.downscale_t(frame)

        if self.corners is None:
            self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self._update_features()

            return np.array([0, 0])

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        u_corners, status, error = cv.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.corners, None, **self.lk_params
        )

        good_new = u_corners[status == 1]
        good_old = self.corners[status == 1]

        global_mov = self._compute_global_movement(good_new, good_old)

        self.old_gray = frame_gray
        self.corners = good_new.reshape(-1, 1, 2)

        self.frame_count += 1
        if self.frame_count % self.update_interval == 0:
            self._update_features()

        return global_mov

    def _update_features(self):
        self.corners = cv.goodFeaturesToTrack(
            self.old_gray, mask=None, **self.feature_params
        )

    def _compute_global_movement(self, good_new, good_old):
        global_mov = np.array([0, 0], dtype=np.float)
        n_mov_vectors = 0
        for new, old in zip(good_new, good_old):
            mov = new - old
            mov_thresh = (
                0.5 / self.downscale_t.factor if self.downscale_t is not None else 0.5
            )
            if np.linalg.norm(mov) < mov_thresh:
                continue

            global_mov += mov
            n_mov_vectors += 1

        global_mov = global_mov / n_mov_vectors if n_mov_vectors else global_mov
        if self.downscale_t:
            global_mov *= self.downscale_t.factor

        return global_mov


class WagonTracker:
    def __init__(self, detector, detection_threshold):
        self.detector = detector
        self.elements_info = SortedDict()
        self.wagons_info = SortedDict()
        self.next_element_id = 0
        self.next_wagon_id = 0
        self.movement_vector = np.array([0.0, 0.0])
        self.detection_threshold = detection_threshold

        self.motion_estimator = MovementEstimator(update_interval=3)

    def __call__(self, image):
        boxes, labels, _ = self.detector(image)

        self._estimate_motion(image)

        self._update_tracking(boxes.numpy(), labels.numpy())
        # print(self.wagons_info)
        return deepcopy(self.elements_info)

    def _estimate_motion(self, image):
        self.movement_vector = self.motion_estimator(image) * 1.7
        self.movement_vector[1] = 0.0

    def _update_tracking(self, boxes, labels):
        if len(self.elements_info) == 0 and len(boxes) > 0:
            self._init_elements_info(boxes, labels)
            self._update_wagons(self.elements_info)
            return

        updated_elements_info, new_elements_info = self._update_elements(boxes, labels)

        notfound_elements_info = self._update_notfound_elements(updated_elements_info)

        updated_elements_info.update(notfound_elements_info)

        n_new_boxes = len(new_elements_info)
        if n_new_boxes > 0:
            new_elements_info = {
                self.next_element_id + id: (box, lbl)
                for id, (box, lbl) in enumerate(new_elements_info)
            }
            updated_elements_info.update(new_elements_info)
            self.next_element_id += n_new_boxes
            self._update_wagons(new_elements_info)

        self.elements_info = updated_elements_info

    def _init_elements_info(self, boxes, labels):
        self.next_element_id = len(boxes)
        self.elements_info = {
            id: (box, lbl) for id, (box, lbl) in enumerate(zip(boxes, labels))
        }

    def _update_elements(self, boxes, labels):
        updated_elements_info = {}

        for t_id, (t_box, t_lbl) in self.elements_info.items():
            if len(boxes) == 0:
                break

            search_mask = labels == t_lbl
            if search_mask.sum() == 0:
                continue

            search_boxes = boxes[search_mask, :]
            search_labels = labels[search_mask]
            search_idxs = np.arange(len(boxes))[search_mask]

            t_box[2:] += self.movement_vector
            t_box[:2] += self.movement_vector
            ious = box_utils.iou_of(t_box, search_boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.0:
                updated_elements_info[t_id] = (
                    search_boxes[n_box_idx],
                    search_labels[n_box_idx],
                )

                boxes = np.delete(boxes, (search_idxs[n_box_idx]), axis=0)
                labels = np.delete(labels, (search_idxs[n_box_idx]), axis=0)

        if len(boxes) > 0:
            new_elements_info = list(zip(boxes, labels))
        else:
            new_elements_info = []

        return updated_elements_info, new_elements_info

    def _update_notfound_elements(self, updated_elements_info: list):
        u_ids = updated_elements_info.keys()

        notfound_elements_info = {}
        for t_id, (t_box, t_lbl) in self.elements_info.items():
            if t_id not in u_ids:
                updated_box = np.copy(t_box)
                if np.linalg.norm(self.movement_vector) > 5:
                    updated_box[2:] = t_box[2:] + self.movement_vector
                    updated_box[:2] = t_box[:2] + self.movement_vector
                notfound_elements_info[t_id] = (updated_box, t_lbl)

        return notfound_elements_info

    def _update_wagons(self, new_elements_info):
        for id, (box, lbl) in new_elements_info.items():
            if lbl != 1:
                continue

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
