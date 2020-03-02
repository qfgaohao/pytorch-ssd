import os
from queue import Queue
from threading import Thread

import cv2 as cv
import numpy as np

from vision.utils import box_utils_numpy as box_utils


class ImageWriter:
    def __init__(
        self,
        video_path,
        queue_sz,
        output_path,
        create_output_folder=True,
        extension='png',
    ):
        self.stopped = False
        self.queue = Queue(maxsize=queue_sz)
        self.frame_count = 0
        self.ext = extension.lower()

        self.out = output_path
        if not os.path.isdir(self.out):
            if create_output_folder:
                os.makedirs(self.out)
            else:
                raise RuntimeError('The output folder is invalid!')

        self.basename_template = (
            os.path.basename(video_path).replace('.', '_')
            + '_frame_{}_box_{}.'
            + self.ext
        )

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            img, boxes, ids = self.queue.get(block=True)
            if boxes is None:
                continue

            for boxid, box in zip(ids, boxes):
                box = box.astype(np.int)

                if box_utils.area_of(box[:2], box[2:]) == 0:
                    continue

                (xmin, ymin, xmax, ymax) = box.astype(np.int)
                wagon = img[ymin:ymax, xmin:xmax, :]
                full_path = os.path.join(
                    self.out, self.basename_template.format(self.frame_count, boxid)
                )
                cv.imwrite(full_path, wagon)

            self.frame_count += 1

    def stop(self):
        self.stopped = True

    def __call__(self, img, boxes, ids):
        self.queue.put((img, boxes, ids), block=True)
