import glob
import sys
import os
import xml.etree.ElementTree as ET
from random import random

def main(filename):
    # ratio to divide up the images
    train = 0.7
    val = 0.2
    test = 0.1
    if (train + test + val) != 1.0:
        print("probabilities must equal 1")
        exit()

    # get the labels
    labels = []
    imgnames = []
    annotations = {}

    with open(filename, 'r') as labelfile:
        label_string = ""
        for line in labelfile:
                label_string += line.rstrip()

    labels = label_string.split(',')
    labels  = [elem.replace(" ", "") for elem in labels]

    # get image names
    for filename in os.listdir("./JPEGImages"):
        if filename.endswith(".jpg"):
            img = filename.rstrip('.jpg')
            imgnames.append(img)

    print("Labels:", labels, "imgcnt:", len(imgnames))

    # initialise annotation list
    for label in labels:
        annotations[label] = []

    # Scan the annotations for the labels
    for img in imgnames:
        annote = "Annotations/" + img + '.xml'
        if os.path.isfile(annote):
            tree = ET.parse(annote)
            root = tree.getroot()
            annote_labels = []
            for labelname in root.findall('*/name'):
                labelname = labelname.text
                annote_labels.append(labelname)
                if labelname in labels:
                    annotations[labelname].append(img)
            annotations[img] = annote_labels
        else:
            print("Missing annotation for ", annote)
            exit() 

    # divvy up the images to the different sets
    sampler = imgnames.copy()
    train_list = []
    val_list = []
    test_list = []

    while len(sampler) > 0:
        dice = random()
        elem = sampler.pop()

        if dice <= test:
            test_list.append(elem)
        elif dice <= (test + val):
            val_list.append(elem)
        else:
            train_list.append(elem) 

    print("Training set:", len(train_list), "validation set:", len(val_list), "test set:", len(test_list))


    # create the dataset files
    create_folder("./ImageSets/Main/")
    with open("./ImageSets/Main/train.txt", 'w') as outfile:
        for name in train_list:
            outfile.write(name + "\n")
    with open("./ImageSets/Main/val.txt", 'w') as outfile:
        for name in val_list:
            outfile.write(name + "\n")
    with open("./ImageSets/Main/trainval.txt", 'w') as outfile:
        for name in train_list:
            outfile.write(name + "\n")
        for name in val_list:
            outfile.write(name + "\n")

    with open("./ImageSets/Main/test.txt", 'w') as outfile:
        for name in test_list:
            outfile.write(name + "\n")

    # create the individiual files for each label
    for label in labels:
        with open("./ImageSets/Main/"+ label +"_train.txt", 'w') as outfile:
            for name in train_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open("./ImageSets/Main/"+ label +"_val.txt", 'w') as outfile:
            for name in val_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")
        with open("./ImageSets/Main/"+ label +"_test.txt", 'w') as outfile:
            for name in test_list:
                if label in annotations[name]:
                    outfile.write(name + " 1\n")
                else:
                    outfile.write(name + " -1\n")

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("usage: python generate_vocdata.py <labelfile>")
        exit()
    main(sys.argv[1])
