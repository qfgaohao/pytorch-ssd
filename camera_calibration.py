import _pickle as pickle
import gzip
import os
import sys
from argparse import ArgumentParser
from glob import iglob

import cv2 as cv
import numpy as np

from wagon_tracking.utils import get_realpath


def get_calibration_coeffs(imgs_dir, pattern_sz=(6, 8)):
    fisheye_flags = (
        cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv.fisheye.CALIB_CHECK_COND
        + cv.fisheye.CALIB_FIX_SKEW
    )
    calibration_flags = (
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
        + cv.CALIB_CB_NORMALIZE_IMAGE
    )
    criteria_flags = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER

    objp = np.zeros((1, pattern_sz[0] * pattern_sz[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : pattern_sz[0], 0 : pattern_sz[1]].T.reshape(-1, 2)

    img_shape = None
    obj_points = []
    img_points = []

    for img_path in iglob(f'{imgs_dir}/*'):
        img = cv.imread(img_path, 0)
        if img is None:
            continue

        print(f'Processing {img_path}... ', end='')

        if img_shape is None:
            img_shape = img.shape
        elif img_shape != img.shape:
            print('SKIP (wrong image size)')

        ret, corners = cv.findChessboardCorners(
            img, patternSize=pattern_sz, flags=calibration_flags
        )

        if ret is True:
            obj_points.append(objp)
            cv.cornerSubPix(img, corners, (3, 3), (-1, -1), (criteria_flags, 30, 0.1))
            img_points.append(corners)

        print('OK')

    n_ok = len(obj_points)
    print(f'Found {n_ok} valid images for calibration.')

    k = np.zeros((3, 3))
    d = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]

    _, _, _, _, _ = cv.fisheye.calibrate(
        obj_points,
        img_points,
        img_shape[::-1],
        k,
        d,
        rvecs,
        tvecs,
        fisheye_flags,
        (criteria_flags, 30, 1e-6),
    )

    return img_shape, k, d


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input-folder',
        type=str,
        required=True,
        help='Path to the calibration images folder.',
    )
    parser.add_argument(
        '-on',
        '--output-name',
        type=str,
        required=False,
        help='The name of the output file containing the camera calibration parameters.',
    )
    parser.add_argument(
        '-of',
        '--output-folder',
        type=str,
        required=False,
        help='Path to the folder where the output, file will be stored.',
    )
    args = parser.parse_args()

    input_folder = get_realpath(args.input_folder)
    if not os.path.isdir(input_folder):
        print('The given input folder is not valid')
        sys.exit(-1)

    if args.output_folder:
        output_folder = get_realpath(args.output_folder)
        if not os.path.isdir(output_folder):
            print('The given output folder is not valid')
            sys.exit(-1)
    else:
        output_folder = get_realpath('./resources')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = 'camera_params'
    output_name += '.pkl.gz'

    output_path = os.path.join(output_folder, output_name)

    img_shape, k, d = get_calibration_coeffs(input_folder)

    camera_params = {'image shape': img_shape[::-1], 'K': k, 'D': d}
    output_file = gzip.open(output_path, 'wb')
    output_file.write(pickle.dumps(camera_params, 2))
    output_file.close()
