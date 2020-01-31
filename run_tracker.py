import os
import sys

import cv2

from wagon_tracking.detection import WagonDetector
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
cap.start()

'''-------------------------- Test code --------------------------'''
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
'''---------------------------------------------------------------'''


detector = WagonDetector(net_type, label_path, model_path, prob_threshold=0.5)
tracker = WagonTracker(detector)

cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

while cap.more():
    orig_image = cap.read()
    if orig_image is None:
        continue

    tracking_info = tracker(orig_image)
    # print(tracking_info)

    # Draw detection boundary
    cv2.line(
        orig_image,
        (frame_width // 2, 0),
        (frame_width // 2, frame_height),
        (0, 0, 255),
        4,
    )

    if tracking_info is not None:
        for id, box, label in tracking_info:
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
            )  # line type
            cv2.circle(orig_image, (int(center[0]), int(center[1])), 3, (0, 0, 255), 3)

    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
