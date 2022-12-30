import cv2


vidcap = cv2.VideoCapture('/home/zeys/Desktop/KOVAN/Ornek2.mp4')
success, image = vidcap.read()
count = 1
while success:
    if count%3000 == 0:
        cv2.imwrite("/home/zeys/Desktop/KOVAN/2/image_%d.jpg" % count, image)
        success, image = vidcap.read()
        print('Saved image ', count)
    count += 1


