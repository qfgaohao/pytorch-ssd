import _pickle as pickle
import gzip

import cv2 as cv
import numpy as np

from wagon_tracking.utils import get_realpath


class DistortionRectifier:
    def __init__(self, params_file):
        params = pickle.load(gzip.open(get_realpath(params_file)))
        self.img_shape = params['image shape']
        self.k = params['K']
        self.d = params['D']
        self.k_optimal = None
        self.k_optimal, _ = cv.getOptimalNewCameraMatrix(
            self.k, self.d, self.img_shape, 0.7, self.img_shape, True
        )
        self.map1 = None
        self.map2 = None
        self.imap1 = None
        self.imap2 = None

        self.map1, self.map2 = cv.fisheye.initUndistortRectifyMap(
            self.k, self.d, None, self.k_optimal, self.img_shape, cv.CV_16SC2
        )

        self.imap1, self.imap2 = self._get_inverse_mappings()

    def _get_inverse_mappings(self):
        w, h = self.img_shape
        yy, xx = np.meshgrid(range(h), range(w), indexing='ij')

        p_vector = np.stack((xx, yy), axis=2).reshape((1, -1, 2)).astype(np.float)

        u_points = cv.fisheye.undistortPoints(
            p_vector, self.k, self.d, P=self.k_optimal
        ).reshape((h, -1, 2))
        map1, map2 = cv.convertMaps(
            u_points.astype(np.float32), None, dstmap1type=cv.CV_16SC2
        )
        return map1, map2

    def __call__(self, image):
        return self.undistort_image(image)

    def undistort_image(self, image):
        return cv.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )

    def distort_image(self, image):
        return cv.remap(
            image,
            self.imap1,
            self.imap2,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
        )

    def distort_points(self, points):
        return tuple(tuple(self.map1[y, x]) for x, y in points)

    def undistort_points(self, points):
        points = np.array([points])
        u_points = cv.fisheye.undistortPoints(points, self.k, self.d, P=self.k_optimal)
        return tuple(tuple(p) for p in np.squeeze(u_points))
