from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.datasets import voc_dataset
import cv2
import sys


model_path = sys.argv[1]
image_path = sys.argv[2]

num_classes = len(voc_dataset.class_names)
net = create_vgg_ssd(num_classes)
net.load(model_path)
predictor = create_vgg_ssd_predictor(net)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 5, 0.4)
print([voc_dataset.class_names[i] for i in labels])
print(probs)
print(boxes)
for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
cv2.imwrite("detected.jpg", image)
