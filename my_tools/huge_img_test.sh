CUDA_VISIBLE_DEVICES=4 python my_tools/huge_img_test.py \
/home/wucx/dataset/tzb/input_path/val/img/ \
work_dirs_tzb/faster_rcnn_orpn_r50_fpn_3x_dota10_mss_tzb_001/img_results/ \
work_dirs_tzb/faster_rcnn_orpn_r50_fpn_3x_dota10_mss_tzb_001/faster_rcnn_orpn_r50_fpn_3x_dota10_mss_tzb.py \
work_dirs_tzb/faster_rcnn_orpn_r50_fpn_3x_dota10_mss_tzb_001/epoch_36.pth \
my_tools/1024_test_sss.json \
--method pyplot
