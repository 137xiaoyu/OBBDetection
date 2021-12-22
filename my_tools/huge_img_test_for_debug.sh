CUDA_VISIBLE_DEVICES=4 python my_tools/huge_img_test.py \
/dev3/fengjq/2grade/ship_class/data/raw_data/test/img/ \
/dev3/fengjq/2grade/OBBDetection/fjq_work_space/img_result/ \
/dev3/fengjq/2grade/OBBDetection/fjq_work_space/faster_rcnn_orpn_r50_fpn_3x_dota10_mss_tzb.py \
/dev3/fengjq/2grade/OBBDetection/fjq_work_space/epoch_36.pth \
my_tools/1024_test_sss.json \
--method pyplot
