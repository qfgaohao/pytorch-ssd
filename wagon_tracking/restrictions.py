import numpy as np

from vision.utils import box_utils_numpy as box_utils


class Restriction:
    def __call__(self, boxes, labels=None):
        if len(boxes) == 0:
            return boxes, labels

        filtered_mask = self.filter_mask(boxes, labels)

        boxes = boxes[filtered_mask, :]
        if labels is not None:
            labels = labels[filtered_mask]

        return boxes, labels

    def filter_mask(self, boxes, labels=None):
        raise NotImplementedError


class ROIRestriction(Restriction):
    def __init__(self, roi, area_threshold=0.5):
        super().__init__()
        roi = np.array(roi)
        xmin, ymin = roi[0::2].min(), roi[1::2].min()
        xmax, ymax = roi[0::2].max(), roi[1::2].max()
        self.roi = np.array([xmin, ymin, xmax, ymax])

        self.area_threshold = area_threshold

    def filter_mask(self, boxes, labels=None):
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

    def filter_mask(self, boxes, labels=None):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2

        num = np.abs(self.a * centers[:, 0] + self.b * centers[:, 1] + self.c)

        boxes_distances = num / self._den
        return boxes_distances <= self.distance_threshold
