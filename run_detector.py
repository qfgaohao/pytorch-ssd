import os
import sys

import cv2

from vision.utils.misc import Timer
from wagon_tracking.detection import WagonDetector
from wagon_tracking.transforms import DistortionRectifier
from wagon_tracking.videostream import VideoFileStream, VideoLiveStream
from wagon_tracking.utils import get_realpath

if len(sys.argv) < 5:
    print(
        'Usage: python run_ssd_example.py <net type>  <model path> <label path>'
        ' <video file | device location> [camera_param_file]'
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

detector = WagonDetector(net_type, label_path, model_path)

cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

timer = Timer()
while cap.more():
    orig_image = cap.read()
    if orig_image is None:
        continue
    timer.start()
    boxes, labels, probs = detector(orig_image)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{detector.class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(
            orig_image,
            label,
            (box[0] + 20, box[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # font scale
            (255, 0, 255),
            2,
        )  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
