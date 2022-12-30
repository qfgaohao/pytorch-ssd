import cv2 as cv
import pandas as pd
import os
# read yolo labels and convert to csv
def read_yolo_labels_and_convert_to_csv(yolo_labels_path):
    cou = 0
    for i in  os.listdir(yolo_labels_path):
        cou += 1
        if i.endswith(".txt"):
            file_name = i.replace('.txt', '')
            img= cv.imread("/home/zeys/Desktop/cropped_image/" + file_name +                    '.jpg')
            image_h, image_w, _ = img.shape
            print(file_name)
            print("image_h,image_w", image_h, image_w)
            with open(yolo_labels_path + i) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    x_center = line[1]
                    y_center = line[2]
                    w = line[3]
                    h = line[4]
                    x1,y1,x2,y2= yolo_to_pascal_voc(x_center, y_center, w, h, image_w, image_h)
                    xmin, ymin, xmax, ymax = open_images_normalization(x1, y1, x2, y2, image_w, image_h)
                    # to check if the conversion is correct
                    # newxmin=xmin*image_w
                    # newymin=ymin*image_h
                    # newxmax=xmax*image_w
                    # newymax=ymax*image_h
                    # cv.rectangle(img, (int(newxmin), int(newymin)), (int(newxmax), int(newymax)), (0, 255, 0), 2)
                    # cv.imshow("img", img)
                    # cv.waitKey(250)
                    new_file_name = file_name + '.jpg'
                    df = pd.DataFrame({'filename': new_file_name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,'ClassName':'traffic_light'}, index=[0])
                    df.to_csv('/home/zeys/Desktop/new_labelsss/real_data.csv', mode='a', header=False, index=True)


# Convert Yolo bb to Pascal_voc bb
def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    ws = float(w) * float(image_w)
    hs = float(h) * float(image_h)
    print("w,h", ws, hs)
    x1 = (2*(float(x_center) * float(image_w)) - float(ws))/2
    y1 = (2*(float(y_center) * float(image_h)) - float(hs))/2
    x2 = float(x1) + float(ws)
    y2 = float(y1) + float(hs)
    return x1, y1, x2, y2

def open_images_normalization(x1, y1, x2, y2, image_w, image_h):
    xmin = x1 / image_w
    ymin = y1 / image_h
    xmax = x2 / image_w
    ymax = y2 / image_h
    return xmin, ymin, xmax, ymax

def add_headers_tod_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['ImageID', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'ClassName']
    df.to_csv(csv_path, index=False)


# read_yolo_labels_and_convert_to_csv('/home/zeys/Desktop/new_labelsss/')
# add_headers_tod_csv('/home/zeys/Desktop/new_labelsss/real_data.csv')
def visualize_real_data(csv_path):
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        img = cv.imread("/media/zeys/PortableSSD/cropped_image/" + row['ImageID'])
        h, w, _ = img.shape
        cv.rectangle(img, (int(row['Xmin']*w), int(row['Ymin']*h)), (int(row['Xmax']*w), int(row['Ymax']*h)), (0, 255, 0), 2)
        cv.imshow("img", img)
        cv.waitKey(200)

visualize_real_data('/media/zeys/PortableSSD/pre_csv_files/bosch-train1.csv')