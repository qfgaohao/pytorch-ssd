import argparse
import cv2


parser = argparse.ArgumentParser(description='Convert video to image')

parser.add_argument("--video_path", type=str, help='Video file path.')
parser.add_argument("--image_save_path", type=str, help='Image save folder.')
def VideoToImage():
    video_path = args.video_path
    image_save_path = args.image_save_path
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    # save every 100 image
    while success:
        if count%100== 0:
            cv2.imwrite(image_save_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
    print("Saved %d images" % count)




if __name__ == '__main__':
    args = parser.parse_args()
    VideoToImage()
