import cv2
import glob
import argparse
import pandas as pd
import os
import random
import csv
parser = argparse.ArgumentParser(description='Dataset Creater')
parser.add_argument('--dataset', type=str, help='Dataset directory path')

def DatasetCreater():
    dataset = args.dataset
    dataset = pd.read_csv(dataset)
    count = 0
    for index, row in dataset.iterrows():
        y_margin = random.randint(0, 2)
        x_margin = random.randint(0, 2)
        file_name = row['ImageID']
        # check if the image is exst
        if os.path.isfile("/media/zeys/PortableSSD/archive/Lisadownsampled/daySequence2/" + file_name):
            xmax = row['Xmax']
            ymax = row['Ymax']
            xmin = row['Xmin']
            ymin = row['Ymin']
            print("readed from csv file: xmax ymax xmin ymin : ", xmax, ymax, xmin, ymin)
            image = cv2.imread("/media/zeys/PortableSSD/archive/Lisadownsampled/daySequence2/" + file_name)
            h, w, _ = image.shape
            print("image h, w: ", h, w)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.imshow("img", image)  # show image
            cv2.waitKey(500)  # wait for 0 milisecond

            x0 = int(xmin - ((xmax - xmin) * x_margin)) if int(xmin - (x_margin * (xmax - xmin))) > 0 else 0
            x1 = int(xmax + ((xmax - xmin) * x_margin)) if int(xmax + (xmax - xmin) * x_margin) < w else w
            y0 = int(ymin + ((ymin - ymax) * y_margin)) if int(ymin + (ymin - ymax) * y_margin) < h else h
            y1 = int(ymax - ((ymin - ymax) * y_margin)) if int(ymax - ((ymin - ymax) * y_margin)) > 0 else 0

            cropped_image = image[y0:y1, x0:x1]
            height, width, _ = cropped_image.shape
            print("cropped image shape : ", height, width)
            if height == 0 or width == 0:
                print("height or width is zero")
                dataset = dataset.drop(index)
                continue
            new_xmin = (xmin - abs(x0)) / width
            new_xmax = (xmax - abs(x0)) / width
            new_ymin = (ymin - abs(y0)) / height
            new_ymax = (ymax - abs(y0)) / height
            print("new xmax ymax xmin ymin : ", new_xmax, new_ymax, new_xmin, new_ymin)

        # double check for scaling
        #     xmax= int(new_xmax * width)
        #     ymax= int(new_ymax * height)
        #     xmin= int(new_xmin * width)
        #     ymin= int(new_ymin * height)
        #     cv2.rectangle(cropped_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        #     cv2.imshow("img", cropped_image)  # show image
        #     cv2.waitKey(500)  # wait for 0 milisecond

        # check if cropped images already saved before
#
            if os.path.exists("/media/zeys/PortableSSD/cropped_image/" + file_name):
                print("Image is  Already Exists")
                file_name = file_name[:-4] + "-" + str(random.randint(1, 100000)) + ".jpg"
                print("new file name : ", file_name)
                cv2.imwrite("/media/zeys/PortableSSD/cropped_image/" + file_name, cropped_image)
            else:
                cv2.imwrite("/media/zeys/PortableSSD/cropped_image/" + file_name, cropped_image)

            df = pd.DataFrame({'ImageID': file_name, 'Xmin': new_xmin, 'Ymin': new_ymin, 'Xmax': new_xmax, 'Ymax': new_ymax,'ClassName':"traffic_light"}, index=[0])
            df.to_csv('/media/zeys/PortableSSD/pre_csv_files/daySequence2.csv', mode='a', header=False, index=True)

if __name__ == '__main__':
    args = parser.parse_args()
    DatasetCreater()