from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot
from mmdet.apis import inference_detector_huge_image
import torch
import mmcv
import os
import BboxToolkit as bt




def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument("--input_dir", default='/input_path', help="input path", type=str)
    parser.add_argument("--output_dir", default='/data/output_path', help="output path", type=str)
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--split', help='split configs in BboxToolkit/tools/split_configs')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    # build the model from a config file and a checkpoint file
    print(args.config)
    print(args.checkpoint)
    print(args.device)

    classes = ['bigship',]
    threshold = args.score_thr



    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    input_img_list = os.listdir(os.path.join(args.input_dir,'img'))
    output_dicts = []
    for input_big_img in input_img_list:
        input_big_img_path = os.path.join(args.input_dir,'img',input_big_img)
        result = inference_detector_huge_image(model, input_big_img_path, args.split, nms_cfg)
        
        output_dict = {}
        output_dict.update({'image_name': input_big_img})
        

        labels = []
        for class_id, bbox_result in enumerate(result):
            if bbox_result.shape[0] != 0:
                for index in range(bbox_result.shape[0]):
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
    output_path = '{}/ship_results.json'.format(args.output_dir)
    print(output_path)
    mmcv.dump(output_dicts, output_path)
    
if __name__ == '__main__':
    main()
