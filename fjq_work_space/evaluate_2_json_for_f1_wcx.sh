python fjq_work_space/evaluate_2_json_for_f1.py \
--label_json fjq_work_space/json/test.json \
--predict_json fjq_work_space/json/ship_results.json \
--visualize True \
--img_input_dir /home/wucx/dataset/tzb/input_path/val/img/ \
--img_output_dir work_dirs_tzb/visualize/fjq_eval_f1

# python fjq_work_space/evaluate_2_json_for_f1.py \
# --label_json fjq_work_space/json/test.json \
# --predict_json fjq_work_space/json/ship_results.json \
# --visualize True \
# --img_input_dir  /dev3/fengjq/2grade/ship_class/data/raw_data/test/img/ \
# --img_output_dir /dev3/fengjq/2grade/ship_class/data/raw_data/test/inference_result/det

# python fjq_work_space/evaluate_2_json_for_f1.py \
# --label_json fjq_work_space/json/test.json \
# --predict_json fjq_work_space/json/ship_results_with_cls.json \
# --visualize True \
# --img_input_dir  /dev3/fengjq/2grade/ship_class/data/raw_data/test/img/ \
# --img_output_dir /dev3/fengjq/2grade/ship_class/data/raw_data/test/inference_result/det_with_cls
