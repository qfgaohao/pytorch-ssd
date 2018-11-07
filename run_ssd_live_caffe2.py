import vision.utils.box_utils_numpy as box_utils
from vision.utils.misc import Timer
from vision.ssd.config.mobilenetv1_ssd_config import specs, center_variance, size_variance


import cv2
import sys
from caffe2.python import core, workspace, net_printer
import numpy as np

priors = box_utils.generate_ssd_priors(specs, 300)
print('priors.shape', priors.shape)


def load_model(init_net_path, predict_net_path):
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()
    p = workspace.Predictor(init_net, predict_net)
    return p


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                  iou_threshold=iou_threshold,
                                  top_k=top_k,
                                  )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


if len(sys.argv) < 2:
    print('Usage: python run_ssd_live_caffe2.py init_net predict_net')
    sys.exit(0)
init_net_path = sys.argv[1]
predict_net_path = sys.argv[2]
label_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]
predictor = load_model(init_net_path, predict_net_path)

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    image = (image - 127) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    timer.start()
    confidences, boxes = predictor.run({'0': image})
    interval = timer.end()
    print('Inference Time: {:.2f}s.'.format(interval))
    timer.start()
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.55)
    interval = timer.end()
    print('NMS Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.shape[0]))
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

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
