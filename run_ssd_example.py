from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.datasets import voc_dataset
from vision.utils.misc import Timer
import cv2
import sys
import numpy as np


if len(sys.argv) < 4:
    print('Usage: python <model path> <nms method> <image Path> [number for the times you test to run.]')
    sys.exit(0)

model_path = sys.argv[1]
nms_method = sys.argv[2]
image_path = sys.argv[3]
output_path = "annotated-output.jpg"


if len(sys.argv) >= 5:
    times = int(sys.argv[4])
else:
    times = 1

num_classes = len(voc_dataset.class_names)
net = create_vgg_ssd(num_classes)
net.load(model_path)
predictor = create_vgg_ssd_predictor(net, candidate_size=200, nms_method=nms_method)
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
timer = Timer()
intervals = []
for i in range(times):
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    intervals.append(timer.end())
intervals = np.array(intervals)
print(f"Ran {len(intervals)} times.Mean time: {intervals.mean()}, Max time: {intervals.max()}, Min time: {intervals.min()}")
print([voc_dataset.class_names[i] for i in labels])
print(probs)
print(boxes)
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    label = voc_dataset.class_names[labels[i]]
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
cv2.imwrite(output_path, orig_image)
