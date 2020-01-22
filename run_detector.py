import sys

import cv2

from vision.utils.misc import Timer
from wagon_tracking.detection import WagonDetector
from wagon_tracking.videostream import VideoFileStream, VideoLiveStream
from wagon_tracking.transforms import DistortionRectifier

if len(sys.argv) < 4:
    print(
        'Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]'
    )
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

transform = [
    DistortionRectifier(
        '/home/camilo/workspace/cv/pytorch-ssd/resources/camera_parameters.pkl.gz'
    )
]
if len(sys.argv) >= 5:
    cap = VideoFileStream(
        sys.argv[4], queue_sz=64, transforms=transform
    )  # capture from file
    frame_time = 1
else:
    cap = VideoLiveStream(transforms=transform)  # capture from camera
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
    boxes, labels, probs = detector.run(orig_image, 10, 0.4)
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
