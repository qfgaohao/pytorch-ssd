from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.datasets import voc_dataset
from vision.utils.misc import Timer
import cv2
import sys


if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <net type>  <model path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]

if len(sys.argv) >= 4:
    cap = cv2.VideoCapture(sys.argv[3])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 640)
    cap.set(4, 480)


num_classes = len(voc_dataset.class_names)
if net_type == "mobilenet-v1-ssd":
    net = create_mobilenetv1_ssd(num_classes)
else:
    net = create_vgg_ssd(num_classes)
net.load(model_path)
if net_type == "mobilenet-v1-ssd":
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = voc_dataset.class_names[labels[i]]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
