import cv2
import json
import random
import pandas as pd

# Store JSON data into json_file

json_data = open('/home/zeys/Downloads/instances_train2014.json')
# print category id from json_data
data = json.load(json_data)
print(data['annotations'][0]['category_id'])
# print(type(data))
# take filename and bounding box from json file
# count = 0
# for i in data['annotations']:
#     # print(i['filename'],i['bndbox'])
#     y_margin = random.randint(0, 2)
#     x_margin = random.randint(0, 2)
#     file_name = i['filename'].replace('train_images\\', '')
#     xmin = int(i['bndbox']['xmin'])
#     ymin = int(i['bndbox']['ymin'])
#     xmax = int(i['bndbox']['xmax'])
#     ymax = int(i['bndbox']['ymax'])
#     # print("file_name,x_min,y_min,x_max,y_max", file_name, xmin, ymin, xmax, ymax)
#     image = cv2.imread("/home/zeys/projects/pytorch-ssd/train_dataset/train_images/" + file_name)
#     h, w, _ = image.shape
#     x0 = int(xmin - ((xmax - xmin) * x_margin)) if int(xmin - (x_margin * (xmax - xmin))) > 0 else 0
#     x1 = int(xmax + ((xmax - xmin) * x_margin)) if int(xmax + (xmax - xmin) * x_margin) < w else w
#     y0 = int(ymin + ((ymin - ymax) * y_margin)) if int(ymin + (ymin - ymax) * y_margin) < h else h
#     y1 = int(ymax - ((ymin - ymax) * y_margin)) if int(ymax - ((ymin - ymax) * y_margin)) > 0 else 0
#     crop_img = image[y0:y1, x0:x1]
#     height, width, _ = crop_img.shape
#     if height == 0 or width == 0:
#         print("height or width is zero")
#         continue
#     new_xmin = (xmin - abs(x0)) / width
#     new_xmax = (xmax - abs(x0)) / width
#     new_ymin = (ymin - abs(y0)) / height
#     new_ymax = (ymax - abs(y0)) / height
#     count += 1
#
#     print("new_xmin,new_ymin,new_xmax,new_ymax", new_xmin, new_ymin, new_xmax, new_ymax)
#     # create file name
#     file_name = file_name.replace('.jpg', '')
#     new_file_name = file_name + '_' + str(count) + '.jpg'
#     cv2.imwrite("/home/zeys/projects/pytorch-ssd/test_dataset2/" + new_file_name, crop_img)
#     df = pd.DataFrame({'filename': new_file_name, 'xmin': new_xmin, 'ymin': new_ymin, 'xmax': new_xmax, 'ymax': new_ymax,'ClassName':"traffic_light"}, index=[0])
#     df.to_csv('/home/zeys/projects/pytorch-ssd/test_dataset2/testtt.csv', mode='a', header=False, index=True)
# read csv file
# df = pd.read_csv('/home/zeys/projects/pytorch-ssd/test_dataset2/testtt.csv')
# print(df.head())
# df.columns = ['ImageID', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'ClassName']
# df.to_csv('/home/zeys/projects/pytorch-ssd/test_dataset2/testtt2.csv', index=False)
