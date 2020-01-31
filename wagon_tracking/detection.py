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
from vision.utils.misc import Timer


class WagonDetector:
    def __init__(self, net_type, label_path, model_path, top_k=10, prob_threshold=0.4):
        self.class_names = ['BACKGROUND'] + [
            name.strip() for name in open(label_path).readlines()
        ]

        self.net = self._create_network(net_type)
        self.net.load(model_path)

        self.predictor = self._create_predictor(net_type)

        self.top_k = top_k
        self.prob_threshold = prob_threshold

        self.timer = Timer()

    def _create_network(self, net_type):
        if net_type == 'vgg16-ssd':
            return create_vgg_ssd(len(self.class_names), is_test=True)
        elif net_type == 'mb1-ssd':
            return create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        elif net_type == 'mb1-ssd-lite':
            return create_mobilenetv1_ssd_lite(len(self.class_names), is_test=True)
        elif net_type == 'mb2-ssd-lite':
            return create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        elif net_type == 'sq-ssd-lite':
            return create_squeezenet_ssd_lite(len(self.class_names), is_test=True)
        else:
            raise RuntimeError(
                "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
            )

    def _create_predictor(self, net_type):
        if net_type == 'vgg16-ssd':
            return create_vgg_ssd_predictor(self.net, candidate_size=200)
        elif net_type == 'mb1-ssd':
            return create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)
        elif net_type == 'mb1-ssd-lite':
            return create_mobilenetv1_ssd_lite_predictor(self.net, candidate_size=200)
        elif net_type == 'mb2-ssd-lite':
            return create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200)
        elif net_type == 'sq-ssd-lite':
            return create_squeezenet_ssd_lite_predictor(self.net, candidate_size=200)
        else:
            raise RuntimeError(
                "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
            )

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.timer.start()
        boxes, labels, probs = self.predictor.predict(
            image, self.top_k, self.prob_threshold
        )
        interval = self.timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        return boxes, labels, probs
