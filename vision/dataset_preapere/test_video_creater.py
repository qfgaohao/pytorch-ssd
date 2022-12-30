import cv2
import numpy as np
import glob
img_array = []

shape = 240, 180
fps = 8
print("img_array",len(img_array))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter("/home/zeys/projects/pytorch-ssd/test_result.avi", fourcc, fps, shape)

for filename in glob.glob('/home/zeys/projects/pytorch-ssd/results/*.jpg'):
    img = cv2.imread(filename)
    resized=cv2.resize(img,shape)
    video.write(resized)


video.release()




