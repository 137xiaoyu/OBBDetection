from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot
from mmdet.apis import inference_detector_huge_image
import torch
import mmcv
import os
import BboxToolkit as bt

from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot
import cv2



def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument("--input_dir", default='/input_path', help="input path", type=str)
    parser.add_argument("--output_dir", default='/data/output_path', help="output path", type=str)
    parser.add_argument('--det_config', help='Config file')
    parser.add_argument('--det_checkpoint', help='Checkpoint file')
    parser.add_argument('--cls_config', help='Config file')
    parser.add_argument('--cls_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--split', help='split configs in BboxToolkit/tools/split_configs')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    # build the model from a config file and a checkpoint file
    # print(args.det_config)
    # print(args.det_checkpoint)
    # print(args.device)

    classes = ['bigship',]
    threshold = args.score_thr
    crop_img_size = 200



    model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    model_cls = init_model(args.cls_config, args.cls_checkpoint, device=args.device)


    # test a single image
    nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    input_img_list = os.listdir(os.path.join(args.input_dir,'img'))
    output_dicts = []
    for input_big_img in input_img_list:
        input_big_img_path = os.path.join(args.input_dir,'img',input_big_img)
        result = inference_detector_huge_image(model, input_big_img_path, args.split, nms_cfg)
        big_img = mmcv.imread(input_big_img_path)
        
        output_dict = {}
        output_dict.update({'image_name': input_big_img})
        

        labels = []
        for class_id, bbox_result in enumerate(result):
            if bbox_result.shape[0] != 0:
                for index in range(bbox_result.shape[0]):
                    # 利用分类的confidence代替原结果
                    # print(0)
                    lt_x,lt_y,rb_x,rb_y = bt.obb2hbb(bbox_result[index,:-1])
                    if lt_x > big_img.shape[1] or lt_y > big_img.shape[0] or rb_x <0 or rb_y<0 :
                        continue
                    lt_x = max(lt_x,0)
                    lt_y = max(lt_y,0)
                    rb_x = min(rb_x,big_img.shape[1])
                    rb_y = min(rb_y,big_img.shape[0])
                    crop_img = big_img[int(lt_y):int(rb_y),int(lt_x):int(rb_x),:]
                    # cv2.imwrite('/dev3/fengjq/2grade/OBBDetection/test.png',crop_img)
                    # print(crop_img.shape)
                    crop_img = cv2.resize(crop_img,(crop_img_size,crop_img_size))
                    result = inference_model(model_cls, crop_img)
                    if result['pred_class'] == 'background':
                        continue
                    else:
                        bbox_result[index, 5] = result['pred_score']
                    if bbox_result[index, 5] > threshold:
                        x1,y1,x2,y2,x3,y3,x4,y4 = bt.obb2poly(bbox_result[index,:-1])
                        points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        category_id = classes[class_id]
                        confidence = bbox_result[index, 5]
                        labels.append({'points': points, 'category_id': category_id, 'confidence': confidence})
        output_dict.update({'labels': labels})
        output_dicts.append(output_dict)
    
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = '{}/ship_results_with_cls.json'.format(args.output_dir)
    print(output_path)
    mmcv.dump(output_dicts, output_path)
    
if __name__ == '__main__':
    main()
