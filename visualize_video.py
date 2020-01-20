from argparse import ArgumentParser
from time import sleep

import cv2 as cv

from wagon_tracking.videostream import VideoStream
from wagon_tracking.distortion import DistortionRectifier


def draw_points(frame, points):
    for p in points:
        cv.circle(frame, p, 20, (0, 255, 0), 5)
        cv.circle(frame, p, 5, (0, 0, 255), 5)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The distorted input video')
    parser.add_argument('-p', '--params-file', type=str, required=True,
                        help='The path to the camera calibration parameters file')
    args = parser.parse_args()

    rectifier = DistortionRectifier(args.params_file)
    img_shape = rectifier.img_shape
    udist_points = ((1255, 765), (1509, 956))
    dist_points = rectifier.distort_points(udist_points)
    rdist_points = rectifier.distort_points(udist_points)

    video_stream = VideoStream(args.input)
    video_fps = video_stream.get(cv.CAP_PROP_FPS)

    video_stream.start()
    sleep(1.0)

    dist_wnd_title = 'Distorted video'
    cv.namedWindow(dist_wnd_title, cv.WINDOW_NORMAL)
    udist_wnd_title = 'Undistorted video'
    cv.namedWindow(udist_wnd_title, cv.WINDOW_NORMAL)
    rdist_wnd_title = 'Redistorted video'
    cv.namedWindow(rdist_wnd_title, cv.WINDOW_NORMAL)

    while video_stream.more():
        dist_frame = video_stream.read()
        tmp_frame = dist_frame.copy()
        draw_points(tmp_frame, dist_points)
        cv.imshow(dist_wnd_title, tmp_frame)

        udist_frame = rectifier(dist_frame)
        tmp_frame = udist_frame.copy()
        draw_points(tmp_frame, udist_points)
        cv.imshow(udist_wnd_title, tmp_frame)

        rdist_frame = rectifier.distort_image(udist_frame)
        tmp_frame = rdist_frame.copy()
        draw_points(tmp_frame, rdist_points)
        cv.imshow(rdist_wnd_title, tmp_frame)

        key = cv.waitKey(int(1/video_fps*1000/4))
        if key == ord('q'):
            break

    cv.destroyAllWindows()
