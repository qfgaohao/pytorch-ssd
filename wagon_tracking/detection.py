import cv2

from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.ssd.mobilenetv1_ssd import (
    create_mobilenetv1_ssd,
    create_mobilenetv1_ssd_predictor,
)
from vision.ssd.mobilenetv1_ssd_lite import (
    create_mobilenetv1_ssd_lite,
    create_mobilenetv1_ssd_lite_predictor,
)
from vision.ssd.squeezenet_ssd_lite import (
    create_squeezenet_ssd_lite,
    create_squeezenet_ssd_lite_predictor,
)
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor


class WagonDetector:
    def __init__(self, net_type, label_path, model_path, top_k=10, prob_threshold=0.4):
        self.class_names = ['BACKGROUND'] + [
            name.strip() for name in open(label_path).readlines()
        ]

        if net_type == 'vgg16-ssd':
            self.net = create_vgg_ssd(len(self.class_names), is_test=True)
        elif net_type == 'mb1-ssd':
            self.net = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        elif net_type == 'mb1-ssd-lite':
            self.net = create_mobilenetv1_ssd_lite(len(self.class_names), is_test=True)
        elif net_type == 'mb2-ssd-lite':
            self.net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        elif net_type == 'sq-ssd-lite':
            self.net = create_squeezenet_ssd_lite(len(self.class_names), is_test=True)
        else:
            raise RuntimeError(
                "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
            )
        self.net.load(model_path)

        if net_type == 'vgg16-ssd':
            self.predictor = create_vgg_ssd_predictor(self.net, candidate_size=200)
        elif net_type == 'mb1-ssd':
            self.predictor = create_mobilenetv1_ssd_predictor(
                self.net, candidate_size=200
            )
        elif net_type == 'mb1-ssd-lite':
            self.predictor = create_mobilenetv1_ssd_lite_predictor(
                self.net, candidate_size=200
            )
        elif net_type == 'mb2-ssd-lite':
            self.predictor = create_mobilenetv2_ssd_lite_predictor(
                self.net, candidate_size=200
            )
        elif net_type == 'sq-ssd-lite':
            self.predictor = create_squeezenet_ssd_lite_predictor(
                self.net, candidate_size=200
            )
        else:
            raise RuntimeError(
                "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
            )

        self.top_k = top_k
        self.prob_threshold = prob_threshold

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predictor.predict(image, self.top_k, self.prob_threshold)
