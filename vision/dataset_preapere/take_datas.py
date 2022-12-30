import cv2 as cv
import os
def DownsampleDataset():
# read images in a folder
    count = 0
    for filename in os.listdir('/media/zeys/PortableSSD/archive/all_images/daySequence2/daySequence2/frames/'):
        count += 1
        if count%4 == 0:
            img = cv.imread('/media/zeys/PortableSSD/archive/all_images/daySequence2/daySequence2/frames/' + filename)
            cv.imwrite('/media/zeys/PortableSSD/archive/Lisadownsampled/daySequence2/' + filename, img)

DownsampleDataset()