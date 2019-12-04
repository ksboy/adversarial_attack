#!/usr/bin/env bash
# train_size =250  sum_batch_size = 32 steps_per_epoch =8 epoch =15 sum_steps= 120
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=512
PREDiCT_BATCH_SIZE=512
MAX_LENGTH=100
EPOCHS=20
LOG_STEPS=1000
WARMUP_STEPS=1000
LR=2e-5
EARLY_STOP=4

for((i=0;i<1;i++));
do
CUDA_VISIBLE_DEVICES=1 python run_glue.py \
    --model_type bert \
    --model_name_or_path /home/mhxia/workspace/BDCI/chinese_wwm_ext_pytorch \
    --do_predict \
    --task_name aa  \
    --data_dir ./data/$i/ \
    --output_dir  /home/mhxia/whou/workspace/my_models/adversarial_attack/$i \
    --max_seq_length $MAX_LENGTH \
    --per_gpu_predict_batch_size $PREDiCT_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --weight_decay 0  ;

done

