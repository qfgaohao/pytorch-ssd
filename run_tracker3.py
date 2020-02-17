import os
import sys

import cv2
import numpy as np

from vision.utils import Timer

from wagon_tracking.detection import WagonDetector
from wagon_tracking.tracking import DetectionAndTrackingTracker
from wagon_tracking.transforms import DistortionRectifier
from wagon_tracking.utils import get_realpath
from wagon_tracking.videostream import VideoFileStream, VideoLiveStream

if len(sys.argv) < 5:
    print(
        'Usage: python run_ssd_example.py <net type>  <model path> <label path>'
        '<video file | device location> [camera_param_file]'
    )
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
video_path = sys.argv[4]
camera_parameters = get_realpath('resources/camera_parameters.pkl.gz')
if len(sys.argv) > 5:
    camera_parameters = get_realpath(sys.argv[5])

transform = [DistortionRectifier(camera_parameters)]
if os.path.exists(video_path):
    cap = VideoFileStream(
        video_path, queue_sz=64, transforms=transform
    )  # capture from file
else:
    cap = VideoLiveStream(video_path, transforms=transform)  # capture from camera
frame_time = int(1 / cap.get(cv2.CAP_PROP_FPS) * 1000)
# cap.stream.set(cv2.CAP_PROP_POS_MSEC, 982000)
cap.start()

detector = WagonDetector(net_type, label_path, model_path, prob_threshold=0.4)
tracker = DetectionAndTrackingTracker(detector, 'csrt', 5)

timer = Timer()

cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

while cap.more():
    timer.start()
    orig_image = cap.read()
    if orig_image is None:
        continue

    tracking_info = tracker(orig_image)

    last_center = None
    if len(tracking_info) != 0:
        for id, (box, label) in tracking_info.items():
            label = f"{detector.class_names[label]}: {id}"
            box = box.astype(np.int)
            cv2.rectangle(
                orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4
            )

            center = (box[2:] + box[:2]) // 2

            cv2.putText(
                orig_image,
                label,
                (int(box[0]), int(box[1] - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2,
            )

            cv2.circle(orig_image, tuple(center), 3, (0, 0, 255), 3)
            cv2.putText(
                orig_image,
                str(center),
                (int(center[0]), int(center[1] + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2,
            )
            if last_center is not None:
                distance = np.linalg.norm(center - last_center)
                cv2.line(
                    orig_image, tuple(center), tuple(last_center), (0, 255, 255), 4
                )
                cv2.putText(
                    orig_image,
                    str(distance),
                    (int(center[0]), int(center[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2,
                )
            last_center = center

    cv2.imshow('annotated', orig_image)

    end_time = timer.end() * 1e3
    wait_time = int(np.clip((frame_time - end_time) / 4, 1, frame_time))
    k = cv2.waitKey(wait_time) & 0xFF
    if k == ord('q') or k == 27:
        break
cv2.destroyAllWindows()
