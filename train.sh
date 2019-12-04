#!/usr/bin/env bash
# train_size =250  sum_batch_size = 32 steps_per_epoch =8 epoch =15 sum_steps= 120
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=512
PREDiCT_BATCH_SIZE=512
MAX_LENGTH=100
EPOCHS=20
LOG_STEPS=500
WARMUP_STEPS=1000
LR=2e-5
EARLY_STOP=4

# for((i=0;i<1;i++));
# do

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_type bert \
    --model_name_or_path /home/mhxia/workspace/BDCI/chinese_wwm_ext_pytorch \
    --do_train \
    --do_eval \
    --eval_all_checkpoints \
    --task_name aa  \
    --data_dir ./data/group/0/ \
    --output_dir  /home/mhxia/whou/workspace/my_models/adversarial_attack/adversarial/0 \
    --max_seq_length $MAX_LENGTH \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE  \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE  \
    --per_gpu_predict_batch_size $PREDiCT_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --warmup_steps $WARMUP_STEPS \
    --adam_epsilon 1e-6 \
    --logging_steps $LOG_STEPS \
    --num_train_epochs $EPOCHS \
    --evaluate_during_training \
    --early_stop $EARLY_STOP \
    --learning_rate $LR \
    --overwrite_output_dir \
    --weight_decay 0  ;
    # --overwrite_output_dir \

# done

