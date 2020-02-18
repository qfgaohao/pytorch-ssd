import os
import sys

import cv2
import numpy as np

from vision.utils import Timer

from wagon_tracking.detection import WagonDetector
from wagon_tracking.restrictions import (
    DetectionDistanceRestriction,
    ROIRestriction,
    TrajectoryProfileRestriction,
)
from wagon_tracking.tracking import WagonTracker
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
if len(sys.argv) > 5:
    camera_parameters = get_realpath(sys.argv[5])
    transform = [DistortionRectifier(camera_parameters)]
else:
    transform = []

if os.path.exists(video_path):
    cap = VideoFileStream(
        video_path, queue_sz=64, transforms=transform
    )  # capture from file
else:
    cap = VideoLiveStream(video_path, transforms=transform)  # capture from camera
frame_time = int(1 / cap.get(cv2.CAP_PROP_FPS) * 1000)
cap.stream.set(cv2.CAP_PROP_POS_MSEC, 150000)
cap.start()

'''-------------------------- Test code --------------------------'''
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
'''---------------------------------------------------------------'''


detector = WagonDetector(net_type, label_path, model_path, prob_threshold=0.4)
restrictions = [
    ROIRestriction((302, 273, 1579, 796)),
    TrajectoryProfileRestriction(
        (0, 0, frame_width, frame_height), (0, frame_height // 2), distance_threshold=20
    ),
    DetectionDistanceRestriction((2.5, 4.8), (0.5, 1.5)),
]
tracker = WagonTracker(
    detector,
    frame_width // 2,
    restrictions=restrictions,
    video_fps=cap.get(cv2.CAP_PROP_FPS),
    target_fps=30.0,
)

timer = Timer()

cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

while cap.more():
    timer.start()
    orig_image = cap.read()
    if orig_image is None:
        continue

    tracking_info = tracker(orig_image)

    # Draw the ROI
    x1, y1, x2, y2 = restrictions[0].roi.tolist()
    cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Draw the trajectory profile
    starting_point, ending_point = restrictions[1].line_points
    xmin, ymin = (int(e) for e in starting_point)
    xmax, ymax = (int(e) for e in ending_point)
    cv2.line(orig_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)

    # Draw detection boundary
    cv2.line(
        orig_image,
        (frame_width // 2, 0),
        (frame_width // 2, frame_height),
        (0, 0, 255),
        4,
    )

    last_center = None
    if len(tracking_info) != 0:
        for id, (box, label) in tracking_info.items():
            label = f"{detector.class_names[label]}: {id}"
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
                    str(distance / 349),
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
