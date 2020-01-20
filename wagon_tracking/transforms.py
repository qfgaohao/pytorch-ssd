import cv2 as cv

from wagon_tracking.distortion import DistortionRectifier


class DistortionRectifierTransform:
    def __init__(self, params_file):
        self.rectifier = DistortionRectifier(params_file)

    def __call__(self, image):
        return self.rectifier.undistort_image(image)


class ImageDownscaleTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        h, w, _ = image.shape
        return cv.resize(image, (int(w // self.factor), int(h // self.factor)))
