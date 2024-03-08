# #!/bin/sh
echo "=========================================================================================="
echo "                                         CCKS                                        "
echo "=========================================================================================="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python src/train.py \
  --task_save_name FLAT_NER \
  --data_dir ./data \
  --data_name ccks2017 \
  --model_name SERS \
  --model_name_or_path /shared_data/pretrained_models/bert-base-chinese \
  --output_dir save_file/ccks2017_2_2_seed42_GAT_LR3E-5_epoch30 \
  --result_dir save_file/ccks2017_2_2_seed42_GAT_LR3E-5_epoch30/result \
  --overwrite_output_dir TRUE \
  --first_label_file ./data/ccks2017/label_map.json \
  --train_set ./data/ccks2017/train.json \
  --dev_set ./data/ccks2017/dev.json \
  --test_set ./data/CMeEE/test.json \
  --use_attn TRUE \
  --seed 42 \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 30 \
  --learning_rate 3e-5 \
  --task_layer_lr 2 \
  --label_str_file ./data/ccks2017/label_annotation.txt \
  --is_chinese TRUE


# #!/bin/sh
# echo "=========================================================================================="
# echo "                                         CMeEE                                        "
# echo "=========================================================================================="
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python src/train.py \
#   --task_save_name FLAT_NER \
#   --data_dir ./data \
#   --data_name CMeEE \
#   --model_name SERS \
#   --model_name_or_path /shared_data/pretrained_models/bert-base-chinese \
#   --output_dir save_file/CMeEE_7_25_ATT_LR2E-5_epoch20 \
#   --result_dir save_file/CMeEE_7_25_ATT_LR2E-5_epoch20/result \
#   --overwrite_output_dir TRUE \
#   --first_label_file ./data/CMeEE/label_map.json \
#   --train_set ./data/CMeEE/train.json \
#   --dev_set ./data/CMeEE/dev.json \
#   --test_set ./data/CMeEE/test.json \
#   --use_attn TRUE \
#   --max_seq_length 512 \
#   --per_gpu_train_batch_size 16 \
#   --gradient_accumulation_steps 2 \
#   --num_train_epochs 20 \
#   --learning_rate 2e-5 \
#   --task_layer_lr 2 \
#   --label_str_file ./data/CMeEE/label_annotation.txt \
#   --is_chinese TRUE