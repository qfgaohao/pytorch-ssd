import cv2
import os
import glob
import pandas as pd
from pathlib import Path
import random as rand

# spilit dataset to train and test set
def split_train_test():
    dataset = pd.read_csv("/media/zeys/PortableSSD/pre_csv_files/total_dataset.csv")
    copy_dataset = dataset.copy()  # copy the df
    whole_dataset = copy_dataset.to_numpy().tolist()
    # imageID ye konrol yapılacak.
    image_id_list = copy_dataset['ImageID'].to_numpy().tolist()
    image_id_set = list(set(image_id_list))
    rand.shuffle(image_id_set)
    train_set_by_imageID = image_id_set[0:22900]  # 70
    validation_set_by_imageID = image_id_set[22900:28900]  # 20
    test_set_by_imageID = image_id_set[28900:]  # 10
    print("train_set_by_imageID", len(train_set_by_imageID))
    print("val_set_by_imageID", len(validation_set_by_imageID))
    print("test_set_by_imageID", len(test_set_by_imageID))
    train_set = []
    test_set = []
    validation_set = []
    print(whole_dataset[0][0])
    # for i in range(len(whole_dataset)):
    #     if whole_dataset[i][0] in train_set_by_imageID:
    #         train_set.append(whole_dataset[i])
    #     elif whole_dataset[i][0] in test_set_by_imageID:
    #         test_set.append(whole_dataset[i])
    #     elif whole_dataset[i][0] in validation_set_by_imageID:
    #         validation_set.append(whole_dataset[i])
    # rand.shuffle(train_set)
    # rand.shuffle(test_set)
    # rand.shuffle(validation_set)
    # # convert list to dataframe
    # train_df = pd.DataFrame(train_set, columns=copy_dataset.columns)
    # test_df = pd.DataFrame(test_set, columns=copy_dataset.columns)
    # validation_df = pd.DataFrame(validation_set, columns=copy_dataset.columns)
    # # save train and test set
    # train_df.to_csv("/media/zeys/PortableSSD/pre_csv_files/train.csv",index=False)
    # validation_df.to_csv("/media/zeys/PortableSSD/pre_csv_files/validation.csv", index=False)
    # test_df.to_csv("/media/zeys/PortableSSD/pre_csv_files/test.csv", index=False)


# take train and test csv and spilit images to two folder train and test
def spilit_images_to_train_test():
    # read a csv data file
    train_dataset = pd.read_csv("/media/zeys/PortableSSD/open_images/train.csv")
    val_dataset = pd.read_csv("/media/zeys/PortableSSD/open_images/validation.csv")
    test_dataset= pd.read_csv("/media/zeys/PortableSSD/open_images/test.csv")
    # iterate over the rows and take Xmax,Ymax,Xmin,Ymin
    for index, row in train_dataset.iterrows():
        file_name = row['ImageID']
        img = cv2.imread("/media/zeys/PortableSSD/cropped_image/" + file_name)  # read image
        # save images to folder
        cv2.imwrite("/media/zeys/PortableSSD/open_images/train/" + file_name, img)
    print("Train images are saved")
    for index, row in val_dataset.iterrows():
        file_name = row['ImageID']
        img = cv2.imread("/media/zeys/PortableSSD/cropped_image/" + file_name)  # read image
        # save images to folder
        cv2.imwrite("/media/zeys/PortableSSD/open_images/validation/" + file_name, img)
    print("Validation images are saved")
    for index, row in test_dataset.iterrows():
        file_name = row['ImageID']
        img = cv2.imread("/media/zeys/PortableSSD/cropped_image/" + file_name)
        # save images to folder
        cv2.imwrite("/media/zeys/PortableSSD/open_images/test/" + file_name, img)
    print("Test images are saved")


# split_train_test()
# delete_unusedname(dataset)
# change_values()
spilit_images_to_train_test()
# change_columns(dataset)
# concanate_images()
# crop_images()
