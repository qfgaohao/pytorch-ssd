import numpy as np

from vision.utils import box_utils_numpy as box_utils


class Restriction:
    def __call__(self, boxes, labels=None, tracking_info=None):
        if len(boxes) == 0:
            return boxes, labels

        filtered_mask = self.filter_mask(boxes, labels, tracking_info)

        boxes = boxes[filtered_mask, :]
        if labels is not None:
            labels = labels[filtered_mask]

        return boxes, labels

    def filter_mask(self, boxes, labels=None, tracking_info=None):
        raise NotImplementedError


class ROIRestriction(Restriction):
    def __init__(self, roi, area_threshold=0.5):
        super().__init__()
        roi = np.array(roi)
        xmin, ymin = roi[0::2].min(), roi[1::2].min()
        xmax, ymax = roi[0::2].max(), roi[1::2].max()
        self.roi = np.array([xmin, ymin, xmax, ymax])

        self.area_threshold = area_threshold

    def filter_mask(self, boxes, labels=None, tracking_info=None):
        total_areas = box_utils.area_of(boxes[:, :2], boxes[:, 2:])

        intersections = np.empty_like(boxes)
        intersections[:, :2] = np.maximum(boxes[:, :2], self.roi[:2])
        intersections[:, 2:] = np.minimum(boxes[:, 2:], self.roi[2:])
        inter_areas = box_utils.area_of(intersections[:, :2], intersections[:, 2:])

        percents_areas = inter_areas.astype(np.float) / total_areas.astype(np.float)

        return percents_areas >= self.area_threshold


class TrajectoryProfileRestriction(Restriction):
    def __init__(self, roi, p_start, p_end=None, distance_threshold=20):
        super().__init__()
        self.points = np.empty((2, 2), dtype=np.float)
        self.points[0, :] = p_start
        self.points[1, :] = p_end if p_end is not None else (1e6, p_start[1])
        self.points.sort(axis=0)

        self.roi = np.array(roi, dtype=np.float)
        x_min, y_min, x_max, y_max = roi
        self.points[:, 0] = np.clip(self.points[:, 0], x_min, x_max)
        self.points[:, 1] = np.clip(self.points[:, 1], y_min, y_max)

        b, neg_a = np.diff(self.points, axis=0)[0]
        c = np.diff(self.points[::-1, 0] * self.points[:, 1])[0]
        self.a = neg_a
        self.b = b
        self.c = c

        self._den = np.sqrt(self.a ** 2 + self.b ** 2)

        self.distance_threshold = distance_threshold

    @property
    def line_points(self):
        starting_point, ending_point = self.points
        return starting_point.tolist(), ending_point.tolist()

    def filter_mask(self, boxes, labels=None, tracking_info=None):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        num = np.abs(self.a * centers[:, 0] + self.b * centers[:, 1] + self.c)

        boxes_distances = num / self._den
        return boxes_distances <= self.distance_threshold


class DetectionDistanceRestriction(Restriction):
    def __init__(self, intrawagon_range, interwagon_range):
        super().__init__()
        self.intrawagon_range = tuple(np.sort(intrawagon_range))
        self.interwagon_range = tuple(np.sort(interwagon_range))
        self.next_class = None

    def filter_mask(self, boxes, labels=None, tracking_info=None):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        heigths = boxes[:, 3] - boxes[:, 1]

        last_key = np.sort(tuple(tracking_info.keys()))[-1]
        last_element = tracking_info[last_key][0]
        last_center = (last_element[:2] + last_element[2:]) / 2
        last_heigth = last_element[3] - last_element[1]

        mask = np.zeros((len(boxes),), dtype=bool)
        for c_idx, (center, heigth) in enumerate(zip(centers, heigths)):
            if center[0] <= last_center[0]:
                continue

            mean_heigth = (heigth + last_heigth) / 2
            length = np.linalg.norm(center - last_center) / mean_heigth
            length_class = self._classify_length(length, mean_heigth)

            last_center = center
            last_heigth = heigth

            if length_class is None:
                continue

            elif self.next_class is None or length_class == self.next_class:
                mask[c_idx] = True
                self.next_class = self._get_next_class(length_class)

        return mask

    def _classify_length(self, length, mean_heigth):
        if self.intrawagon_range[0] <= length <= self.intrawagon_range[1]:
            return 0
        elif self.interwagon_range[0] <= length <= self.interwagon_range[1]:
            return 1
        else:
            return None

    def _get_next_class(self, length_class):
        if length_class == 0:
            return 1

        return 0
