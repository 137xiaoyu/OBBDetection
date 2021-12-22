from ast import parse
import os
import json
import numpy as np
from shapely.geometry import Polygon
# import ipdb
import cv2
classes = ['bigship', ]
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from BboxToolkit.geometry import bbox_overlaps

def intersection(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def eval_f1_oneclass(gt, det, thres):
    tp = 0
    fp = 0
    fn = 0

    rows = len(det)
    cols = len(gt)

    matrix = np.zeros((rows, cols))
    cur_max_idx = -1    # for each row, choose max iou to set 1 and set other iou>0.5 to 0
    cur_max_val = 0
    for i in range(rows):
        for j in range(cols):
            
            # print(1)
            # iou = intersection(det[i], gt[j])
            # itertools.chain.from_iterable(det[i])
            iou = bbox_overlaps(np.array(det[i]).reshape(1,-1), np.array(gt[j]).reshape(1,-1))
            iou = iou[0][0]
            # print(iou)

            if iou < thres:
                iou = 0
            else:
                if iou > cur_max_val and cur_max_idx != -1:
                    matrix[i,cur_max_idx] = 0
                    cur_max_idx = j
                    cur_max_val = iou
                iou = 1
            matrix[i,j] = iou

    # calculate col values
    colsum = np.sum(matrix, axis=0)
    for j in range(cols):
        if colsum[j] == 0:
            fn += 1         # gt not match any det => false negative
        elif colsum[j] == 1:
            tp += 1         # corret match => true positive
        elif colsum[j] > 1:
            tp += 1
            fp += colsum[j] - 1     # more than one det match a gt => 1 tp and (n-1) fp
        else:
            print('wrong value!')

    # calculate row values
    rowsum = np.sum(matrix, axis=1)
    for i in range(rows):
        if rowsum[i] == 0:
            fp += 1         # det not match any gt => false positive

    return tp, fp, fn


def evaluate_two_jsons(label_json,predict_json,thres=0.5):
    with open(label_json,'r') as f:
        label_dict = json.load(f)
        if len(label_dict)==0:
            print('label_json is nothing!')
            exit(0)
    with open(predict_json,'r') as f:
        predict_dict = json.load(f)
        if len(predict_dict) == 0:
            print('predict_json is nothing!')
            exit(0)
    # compare two json
    # print(0)
    img_name_list_in_predict_dict = [img_info['image_name'] for img_info in predict_dict]

    # caculate tp tn fp fn class by class
    precision =[]
    recall = []
    F1 = []
    for class_name in classes:
        tp_class = 0
        fp_class = 0
        fn_class = 0
        for i ,img_info, in enumerate(label_dict):
            img_name = img_info['image_name']
            # search the img in predict_dict
            id = img_name_list_in_predict_dict.index(img_name)
            label_per_img = img_info['labels']
            predict_per_img = predict_dict[id]['labels']

            bbox_of_label = [label_of_instence\
                             for label_of_instence in label_per_img if label_of_instence['category_id'] == class_name]
            bbox_of_predict = [predict_of_instence\
                             for predict_of_instence in predict_per_img if predict_of_instence['category_id'] == class_name]
            gt = [bbox['points'] for bbox in bbox_of_label]
            det = [bbox['points'] for bbox in bbox_of_predict]
            # ipdb.set_trace()
            tp, fp, fn = eval_f1_oneclass(gt, det, thres)
            tp_class = tp_class + tp
            fp_class = fp_class + fp
            fn_class = fn_class + fn
            # print(0)
        precision.append(tp_class/(tp_class+fp_class))
        recall.append(tp_class/(tp_class+fn_class))
        F1.append(2*tp_class/(2*tp_class+fp_class+fn_class))
    return precision,recall,F1


def evaluate_two_jsons_with_different_confidence(label_json,predict_json,confidence,thres=0.5):
    with open(label_json,'r') as f:
        label_dict = json.load(f)
        if len(label_dict)==0:
            print('label_json is nothing!')
            exit(0)
    with open(predict_json,'r') as f:
        predict_dict = json.load(f)
        if len(predict_dict) == 0:
            print('predict_json is nothing!')
            exit(0)
    # compare two json
    # print(0)
    img_name_list_in_predict_dict = [img_info['image_name'] for img_info in predict_dict]

    # caculate tp tn fp fn class by class
    precision =[]
    recall = []
    F1 = []
    for class_name in classes:
        tp_class = 0
        fp_class = 0
        fn_class = 0
        for i ,img_info, in enumerate(label_dict):
            img_name = img_info['image_name']
            # search the img in predict_dict
            id = img_name_list_in_predict_dict.index(img_name)
            label_per_img = img_info['labels']
            predict_per_img = predict_dict[id]['labels']

            bbox_of_label = [label_of_instence\
                             for label_of_instence in label_per_img if label_of_instence['category_id'] == class_name]
            bbox_of_predict = [predict_of_instence\
                             for predict_of_instence in predict_per_img if predict_of_instence['category_id'] == class_name]
            gt = [bbox['points'] for bbox in bbox_of_label ]
            det = [bbox['points'] for bbox in bbox_of_predict if bbox['confidence'] >= confidence]
            # ipdb.set_trace()
            tp, fp, fn = eval_f1_oneclass(gt, det, thres)
            tp_class = tp_class + tp
            fp_class = fp_class + fp
            fn_class = fn_class + fn
            # print(0)
        if tp_class+fp_class==0:
            precision.append(0)
        else:
            precision.append(tp_class / (tp_class + fp_class))
        if tp_class+fp_class==0:
            recall.append(0)
        else:
            recall.append(tp_class/(tp_class+fn_class))
        if 2*tp_class+fp_class+fn_class==0:
            F1.appedn(0)
        else:
            F1.append(2*tp_class/(2*tp_class+fp_class+fn_class))
    return precision,recall,F1




if __name__ =="__main__":

    parser = ArgumentParser()
    parser.add_argument("--label_json",default="/dev3/fengjq/2grade/ship_class/data/raw_data/test/test.json")
    parser.add_argument("--predict_json",default="/dev3/fengjq/2grade/ship_class/data/raw_data/test/ship_results.json")
    parser.add_argument("--visual",default=None)
    parser.add_argument("--img_input_dir",default='/dev3/fengjq/2grade/ship_class/data/raw_data/test/img/')
    parser.add_argument("--img_output_dir",default='/dev3/fengjq/2grade/ship_class/data/raw_data/test/inference_result/')

    args = parser.parse_args()
    
    label_json = args.label_json
    predict_json = args.predict_json

    # TODO: 增加可视化结果
    if args.visual != None:
        
        print("visualization")



    # 计算评价指标
    precision_all = []
    recall_all = []
    F1_all = []
    for i in np.linspace(0.5,0.95,10):
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json, predict_json,i,0.5)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)
    
    
    mF1 = np.mean(F1_all)
    print("confidence   precision   recall  F1  ")
    precision_all.pop()
    recall_all.pop()
    F1_all.pop()
    for i in np.linspace(0.91,0.99,9):
        precision, recall, F1 = evaluate_two_jsons_with_different_confidence(label_json, predict_json,i,0.5)
        precision_all.append(precision)
        recall_all.append(recall)
        F1_all.append(F1)
    
    confidence_list =np.append(np.linspace(0.5,0.90,9),np.linspace(0.91,0.99,9))
    for i,(p,r,f1) in enumerate(zip(precision_all,recall_all,F1_all)):
        print('{:.2f}         {:.2f}        {:.2f}    {:.2f}'.format(confidence_list[i],p[0],r[0],f1[0]))
    print("mF1 = {}".format(mF1))

    