import argparse
import cv2
import json
import pandas as pd
from csv_preaper import add_headers_tod_csv

def convert_coco_json_to_csv(filename):
    # COCO2017/annotations/instances_val2017.json
    s = json.load(open(filename, 'r'))
    out_file = filename[:-5] + '.csv'
    out = open(out_file, 'w')
    out.write('ImageID,ClassName,Xmin,Xmax,Ymin,Ymax,label\n')

    for ann in s['annotations']:
        if ann['category_id'] == 10:
            for img in s['images']:
                if img['id'] == ann['image_id']:
                    x1 = ann['bbox'][0]
                    x2 = ann['bbox'][0] + ann['bbox'][2]
                    y1 = ann['bbox'][1]
                    y2 = ann['bbox'][1] + ann['bbox'][3]
                    out.write('{},{},{},{},{},{},{}\n'.format(ann['image_id'], img['file_name'], x1, x2, y1, y2,
                                                              ann['category_id']))
    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.to_csv("/home/zeys/projects/pytorch-ssd/vision/dataset_preapere/coco.csv", index=False)

convert_coco_json_to_csv("/home/zeys/Downloads/instances_train2014.json")