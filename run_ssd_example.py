from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.datasets import voc_dataset
from vision.utils.misc import Timer
import cv2
import sys


if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <net type>  <model path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
image_path = sys.argv[3]

num_classes = len(voc_dataset.class_names)
if net_type == "mobilenet-v1-ssd":
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
else:
    net = create_vgg_ssd(num_classes, is_test=True)
net.load(model_path)
if net_type == "mobilenet-v1-ssd":
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
