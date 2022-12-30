
import albumentations as A
import cv2
import os
import pandas as pd
import numpy as np
# declare augmentation pipeline

def DataAugmentation(image_folder,csv_file):
    dataset = pd.read_csv(csv_file)
    aug = A.Compose([
        A.HorizontalFlip(p=1),
        # A.RandomBrightnessContrast(p=1),
    ])

# iterate over each row
    for index, row in dataset.iterrows():
        file_name = row['ImageID']
        image = cv2.imread(image_folder + file_name)
        augmented = aug(image=image)
        flipped_image= augmented['image']
        xmin = 1-row['Xmin']
        ymin =1- row['Ymin']
        xmax = row['Xmax']
        ymax = row['Ymax']
        class_name = row['ClassName']

        # verti = np.concatenate((image, flipped_image), axis=1)
        # cv2.imshow("img", verti)  # show image
        # cv2.waitKey(500)  # wait for 0 milisecond

        new_image_name = file_name.replace('.jpg', '_aug_flipped.jpg')
        # print("new image name : ", new_image_name)

        # h, w, _ = image.shape
        # print("image h, w: ", h, w)
        # cv2.rectangle(image, (int(row['Xmin']), int(row['Ymin'])), (int(row['Xmax']), int(row['Ymax'])), (0, 255, 0), 2)
        # cv2.imshow("img", image)
        #


























# #     read csv file
#     csv_file = pd.read_csv(csv_file)
#     image_names = csv_file['ImageID']
#     xmax= csv_file['Xmax']
#     xmin = csv_file['Xmin']
#     ymax = csv_file['Ymax']
#     ymin = csv_file['Ymin']
#     class_name = csv_file['ClassName']
#
# # read images by image names
#     for image_name in image_names:
#         image_path = os.path.join(image_folder,image_name)
#         image = cv2.imread(image_path)
#         aug = A.Compose([
#             A.HorizontalFlip(p=1),
#             # A.RandomBrightnessContrast(p=1),
#         ])
#         augmented = aug(image=image)
#         image= augmented['image']
#         new_image_name = image_name.replace('.jpg', '_aug_flipped.jpg')
#         print("new image name : ", new_image_name)
#         cv2.imwrite("/media/zeys/PortableSSD/reall_data/augmented_data_flipped/" + new_image_name, image)
# #         save newimage name xmax xmin ymax ymin class name to csv
#         new_csv_file = pd.DataFrame({'ImageID': new_image_name, 'Xmin': xmin, 'Ymin': ymin, 'Xmax': xmax, 'Ymax': ymax,'ClassName':class_name}, index=[0])
#         new_csv_file.to_csv('/media/zeys/PortableSSD/reall_data/real_data_aug_flipped.csv', mode='a', header=False, index=True)
#
DataAugmentation("/media/zeys/PortableSSD/reall_data/cropped_image/","/media/zeys/PortableSSD/pre_csv_files/real_data.csv")