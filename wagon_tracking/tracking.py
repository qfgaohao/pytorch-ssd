from copy import deepcopy

import cv2 as cv
import numpy as np
from sortedcontainers import SortedDict

import vision.utils.box_utils_numpy as box_utils

from wagon_tracking.transforms import ImageDownscaleTransform


class OpticalMovementEstimator:
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
            global_mov *= self.downscale_t.factor ** 2

        return global_mov


class BoxesMovementEstimator:
    def __init__(self):
        self.movement = np.array([0, 0])

    def __call__(self, last_positions, new_positions):
        last_centers = (last_positions[:, :2] + last_positions[:, 2:]) / 2
        new_centers = (new_positions[:, :2] + new_positions[:, 2:]) / 2

        move_vectors = new_centers - last_centers
        return move_vectors.mean(axis=0)


class WagonTracker:
    def __init__(
        self,
        detector,
        detection_threshold,
        video_fps=30,
        target_fps=30,
        restrictions=[],
    ):
        self.detector = detector
        self.elements_info = SortedDict()
        self.next_element_id = 0
        self.optical_movement = np.array([0.0, 0.0])
        self.boxes_movement = np.array([0.0, 0.0])
        self.detection_threshold = detection_threshold
        self.restrictions = restrictions

        self.optical_motion_estimator = OpticalMovementEstimator(update_interval=3)
        self.boxes_motion_estimator = BoxesMovementEstimator()

        self.video_fps = video_fps
        self.target_fps = target_fps
        self.fps_ratio = self.target_fps / self.video_fps

    def __call__(self, image):
        boxes, labels, _ = self.detector(image)
        boxes, labels = boxes.numpy(), labels.numpy()

        if len(boxes) != 0:
            boxes, labels = self._sort_detections(boxes, labels)

        self._estimate_optical_motion(image)

        self._update_tracking(boxes, labels)

        return deepcopy(self.elements_info)

    def _sort_detections(self, boxes, labels):
        sorted_idxs = np.argsort(boxes[:, 0], axis=0)
        return boxes[sorted_idxs, :], labels[sorted_idxs]

    def _estimate_optical_motion(self, image):
        self.optical_movement = self.optical_motion_estimator(image).astype(np.float)
        self.optical_movement *= self.fps_ratio
        self.optical_movement[1] = 0.0
        self.optical_movement[0] = np.clip(self.optical_movement[0], -45.0, 45.0)

    def _estimate_boxes_motion(self, updated_elements_info):
        last_positions = []
        new_positions = []
        for key in updated_elements_info.keys():
            last_positions.append(self.elements_info[key][0])
            new_positions.append(updated_elements_info[key][0])

        if len(last_positions) == 0 or len(new_positions) == 0:
            if (self.optical_movement != 0).any() and (self.boxes_movement == 0).all():
                self.boxes_movement = self.optical_movement
            return

        last_positions = np.asarray(last_positions)
        new_positions = np.asarray(new_positions)
        boxes_movement = self.boxes_motion_estimator(last_positions, new_positions)
        boxes_movement[1] = 0.0
        boxes_movement[0] = np.clip(boxes_movement[0], -45.0, 45.0)

        if np.linalg.norm(boxes_movement) < 3:
            boxes_movement = np.zeros_like(boxes_movement)
        self.boxes_movement = (boxes_movement + self.boxes_movement) / 2

    def _update_tracking(self, boxes, labels):
        if len(self.elements_info) == 0 and len(boxes) > 0:
            self._init_elements_info(boxes, labels)
            return

        updated_elements_info, remaining_elements_info = self._update_elements(
            boxes, labels
        )

        self._estimate_boxes_motion(updated_elements_info)

        notfound_elements_info = self._update_notfound_elements(updated_elements_info)
        updated_elements_info.update(notfound_elements_info)

        for restriction in self.restrictions:
            remaining_elements_info = restriction(
                *remaining_elements_info, updated_elements_info
            )

        new_elements_info = self._get_new_elements_info(
            remaining_elements_info, updated_elements_info
        )

        if len(new_elements_info) > 0:
            updated_elements_info.update(new_elements_info)

        updated_elements_info = self._check_elements_tracking_info(
            updated_elements_info
        )

        self.elements_info = updated_elements_info

    def _init_elements_info(self, boxes, labels):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        # Elements in the left side of the threshold line.
        self.elements_info = {
            id: (box, lbl)
            for id, (box, lbl, cen) in enumerate(zip(boxes, labels, centers))
            if cen[0] < self.detection_threshold and id < 2
        }

        # Elements in the right side of the threshold line.
        self.next_element_id = len(self.elements_info)
        self.elements_info.update(
            {
                id + self.next_element_id: (box, lbl)
                for id, (box, lbl, cen) in enumerate(zip(boxes, labels, centers))
                if cen[0] >= self.detection_threshold and id < 2
            }
        )
        self.next_element_id = len(self.elements_info)

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

            t_box = t_box.copy()
            t_box[2:] += self.optical_movement
            t_box[:2] += self.optical_movement
            ious = box_utils.iou_of(t_box, search_boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.1:
                updated_elements_info[t_id] = (
                    search_boxes[n_box_idx],
                    search_labels[n_box_idx],
                )

                boxes = np.delete(boxes, (search_idxs[n_box_idx]), axis=0)
                labels = np.delete(labels, (search_idxs[n_box_idx]), axis=0)

        return updated_elements_info, (boxes, labels)

    def _update_notfound_elements(self, updated_elements_info: list):
        u_ids = updated_elements_info.keys()

        notfound_elements_info = {}
        for t_id, (t_box, t_lbl) in self.elements_info.items():
            if t_id not in u_ids:
                updated_box = np.copy(t_box)
                updated_box[2:] = t_box[2:] + self.boxes_movement
                updated_box[:2] = t_box[:2] + self.boxes_movement
                notfound_elements_info[t_id] = (updated_box, t_lbl)

        return notfound_elements_info

    def _get_nvisible_elements(self, updated_elements_info):
        visible_left, visible_right = 0, 0

        for id, (box, lbl) in updated_elements_info.items():
            if lbl != 1:
                continue

            if ((box[:2] + box[2:]) / 2)[0] < self.detection_threshold:
                visible_left += 1
            else:
                visible_right += 1

        return visible_left, visible_right

    def _get_new_elements_info(self, remaining_elements_info, updated_elements_info):
        (boxes, labels) = remaining_elements_info

        if len(boxes) == 0:
            return {}

        new_elements_info = {
            id + self.next_element_id: (box, lbl)
            for id, (box, lbl) in enumerate(zip(boxes, labels))
        }
        self.next_element_id += len(new_elements_info)

        return new_elements_info

    def _check_elements_tracking_info(self, elements_info):
        elements_info = SortedDict(elements_info)
        n_elements = len(elements_info)

        keys = elements_info.keys()
        for idx in range(n_elements):
            if idx == n_elements - 1:
                break

            cur_key = keys[idx]
            next_key = keys[idx + 1]

            if elements_info[next_key][0][0] <= elements_info[cur_key][0][0]:
                tmp = elements_info[cur_key]
                elements_info[cur_key] = elements_info[next_key]
                elements_info[next_key] = tmp

        return elements_info
