from queue import Queue
from threading import Thread
from time import sleep

import cv2 as cv

from wagon_tracking.utils import get_realpath, warning


class VideoStreamBase:
    def __init__(self, queue_sz, transforms=[]):
        self.stream = None
        self.stopped = False
        self.queue = Queue(maxsize=queue_sz)
        self.video_frames_count = 0
        self.transforms = transforms

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        raise NotImplementedError()

    def read(self):
        raise NotImplementedError()

    def more(self):
        raise NotImplementedError()

    def stop(self):
        self.stopped = True

    def get(self, prop):
        return self.stream.get(prop)


class VideoFileStream(VideoStreamBase):
    def __init__(self, video_file, queue_sz=128, transforms=[]):
        super().__init__(queue_sz=queue_sz, transforms=transforms)
        self.stream = cv.VideoCapture(get_realpath(video_file))
        self.total_video_frames = int(self.stream.get(cv.CAP_PROP_FRAME_COUNT))
        if self.total_video_frames <= 0:
            warning(
                'WARNING: Could not get the total number of frames of the video. The file may be corrupted!'
            )

    def update(self):
        while True:
            if self.stopped:
                return

            grabbed, frame = self.stream.read()

            if not grabbed:
                self.stop()
                return

            self.queue.put(frame, block=True)

    def read(self):
        self.video_frames_count += 1
        frame = self.queue.get(block=True)
        for transform in self.transforms:
            frame = transform(frame)
        return frame

    def more(self):
        if self.queue.qsize() > 0:
            return True

        elif self.video_frames_count < self.total_video_frames:
            sleep(1.0)
            return self.queue.qsize() > 0

        else:
            return False


class VideoLiveStream(VideoStreamBase):
    def __init__(self, queue_sz=3, transforms=[]):
        super().__init__(queue_sz=queue_sz, transforms=transforms)
        self.stream = cv.VideoCapture(0)
        self.wait_time = 1 / self.stream.get(cv.CAP_PROP_FPS) / 2

    def update(self):
        while True:
            if self.stopped:
                return

            grabbed, frame = self.stream.read()

            if not grabbed:
                self.stop()
                return

            self.queue.put(frame, block=True)

    def read(self):
        self.video_frames_count += 1
        frame = self.queue.get()
        for transform in self.transforms:
            frame = transform(frame)
        return frame

    def more(self):
        if self.queue.qsize() > 0:
            return True
