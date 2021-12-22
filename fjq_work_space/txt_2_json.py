import argparse
import os
import json
import numpy as np
from shapely.geometry import Polygon
# import ipdb
import cv2
classes = ['bigship', ]
import matplotlib.pyplot as plt

from argparse import ArgumentParser

def mask_to_json(mask_dir,json_dir,json_name):
    '''
        mask_dir : the dir of tzb data mask  ,labels in mask are stored in txt, which form is like:
        bigship,x1,y1,x2,y2,x3,y3,x4,y4

        the function  transform the txt to json,the labels in json are like:
        [
            {
            "image_name":"name1.png" ,
            "labels":[
              {
                 "category_id":"bigship" ,
                 "points":[
                    [
                       11.111111111111,
                       11.111111111111
                    ],
                    [
                       22.222222222222,
                       11.111111111111
                    ],
                    [
                       22.222222222222,
                       22.222222222222
                    ],
                    [
                       11.111111111111,
                       22.222222222222
                    ]
                       ],
                 "confidence":0.99
              }]
            }
        ]
    '''
    mask_names = os.listdir(mask_dir)
    default_confidence = 1
    # json content
    json_list = []
    for name in mask_names:
        img_dict ={}
        pic_name = name.split('.')[0] + '.png'
        # json_dict.update({'image_name': pic_name})
        img_dict['image_name'] = pic_name
        label_list = []
        with open(os.path.join(mask_dir,name),'r') as f:
            # make a dict for a mask
            line = f.readline()
            while line:
                temp_label_dict = {}
                info = line.splitlines()[0].split(',')
                info[1:] = [round(float(i),5) for i in info[1:]]
                target_name = info[0]
                temp_label_dict.update({"category_id": target_name})
                points = [
                    [
                        info[1],
                        info[2]
                    ],
                    [
                        info[3],
                        info[4]
                    ],
                    [
                        info[5],
                        info[6]
                    ],
                    [
                        info[7],
                        info[8]
                    ],
                ]
                temp_label_dict.update({"points": points})
                temp_label_dict.update({"confidence": default_confidence})
                label_list.append(temp_label_dict)
                line = f.readline()
        img_dict['labels'] = label_list
        json_list.append(img_dict)
    # write the json_list to .json
    classes = ['bigship',]
    for class_name in classes:
        with open(os.path.join(json_dir,json_name),'w') as f:
            json.dump(json_list,f,indent=1)
    return os.path.join(json_dir,json_name)


if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--txt_dir",help="txt anotation dir",default="/dev3/fengjq/2grade/ship_class/data/raw_data/train/mask/")
    parser.add_argument("--output_json_dir",help="output .json dir",default="/dev3/fengjq/2grade/ship_class/data/raw_data/train/")
    parser.add_argument("--json_name",help="output json name",default="ship.json")
    args = parser.parse_args()
    
    # original_dir = './fjq_workspace/input_data/Data_root/val/labelTxt/'
    # json_dir = './fjq_workspace/input_data/Data_root/val/'


    original_dir = args.txt_dir
    json_dir = args.output_json_dir
    json_name = args.json_name
    json_out = mask_to_json(original_dir,json_dir,json_name)
    print(json_out)
    print("done!")
