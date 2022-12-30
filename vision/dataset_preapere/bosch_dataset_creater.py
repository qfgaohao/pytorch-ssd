import cv2
import pandas as pd
import random
import yaml
from yaml.loader import SafeLoader

# with open('/media/zeys/PortableSSD/Zeynep/bosch_train_dataset/train.yaml') as f:
#     data = yaml.load(f, Loader=SafeLoader)
#
# # read boxes inside data
# for i in data:
#         y_margin = random.randint(0, 2)
#         x_margin = random.randint(0, 2)
#         path = i['path'].split('/')[-1]
#         for box in i['boxes']:
#             xmin = int(box['x_min'])
#             ymin = int(box['y_min'])
#             xmax = int(box['x_max'])
#             ymax = int(box['y_max'])
#             label=box['label']
#             image = cv2.imread("/media/zeys/PortableSSD/Zeynep/bosch_train_dataset/rgb/train/all_data/" + path)
#             h, w, _ = image.shape
#             x0 = int(xmin - ((xmax - xmin) * x_margin)) if int(xmin - (x_margin * (xmax - xmin))) > 0 else 0
#             x1 = int(xmax + ((xmax - xmin) * x_margin)) if int(xmax + (xmax - xmin) * x_margin) < w else w
#             y0 = int(ymin + ((ymin - ymax) * y_margin)) if int(ymin + (ymin - ymax) * y_margin) < h else h
#             y1 = int(ymax - ((ymin - ymax) * y_margin)) if int(ymax - ((ymin - ymax) * y_margin)) > 0 else 0
#
#             crop_img = image[y0:y1, x0:x1]
#             height, width, _ = crop_img.shape
#             if height == 0 or width == 0:
#                 print("height or width is zero")
#                 continue
#             new_xmin = (xmin - abs(x0)) / width
#             new_xmax = (xmax - abs(x0)) / width
#             new_ymin = (ymin - abs(y0)) / height
#             new_ymax = (ymax - abs(y0)) / height

            # xmax= int(new_xmax * width)
            # ymax= int(new_ymax * height)
            # xmin= int(new_xmin * width)
            # ymin= int(new_ymin * height)
            # cv2.rectangle(crop_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            # cv2.imshow("img", crop_img)  # show image
            # cv2.waitKey(5)  # wait for 0 milisecond
            # val=random.randint(0, 100000)
            # print("new_xmin,new_ymin,new_xmax,new_ymax", new_xmin, new_ymin, new_xmax, new_ymax)
            # file_name = path.replace('.png', '')
            # new_file_name = file_name + '_' + str(val) + '.jpg'
            # cv2.imwrite("/media/zeys/PortableSSD/open_images/train/" + new_file_name, crop_img)
            #
            #
            # df = pd.DataFrame({'filename': new_file_name, 'xmin': new_xmin, 'ymin': new_ymin, 'xmax': new_xmax, 'ymax': new_ymax,'ClassName':"traffic_light"}, index=[0])
            # df.to_csv('/media/zeys/PortableSSD/open_images/bosch-train.csv', mode='a', header=False, index=True)

def add_headers_tod_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = ['ImageID', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'ClassName']
    df.to_csv(csv_path, index=False)


# add_headers_tod_csv('/home/zeys/Desktop/new_labelsss/bag1.csv')


def visualize_bosch_dataset(csv_path):
    count=0
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        img = cv2.imread("/media/zeys/PortableSSD/reall_data/augmented_data_flipped/" + row['ImageID'])
        h, w, _ = img.shape
        cv2.rectangle(img, (int(row['Xmin']*w), int(row['Ymin']*h)), (int(row['Xmax']*w), int(row['Ymax']*h)), (0, 255, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(1000)
visualize_bosch_dataset('/media/zeys/PortableSSD/reall_data/real_data_aug_flipped.csv')