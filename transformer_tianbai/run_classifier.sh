#! /usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export GLUE_DIR=/disc2/tianbaili/fenlei

python -m torch.distributed.launch \
    --nproc_per_node 4 ./examples/run_classifier.py \
    --model_type bert \
    --model_name_or_path /disc2/tianbaili/transformers/classifier_outputs/ernie \
    --task_name sn-cp-canshu \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/suning_canshu \
    --train_file train_canshu_trim.csv \
    --predict_file test_canshu_trim.csv \
    --max_seq_length 123 \
    --per_gpu_train_batch_size 6 \
    --per_gpu_eval_batch_size 6 \
    --learning_rate 2e-5 \
    --num_train_epochs 12 \
    --tokenizer_name /disc2/tianbaili/transformers/classifier_outputs/ernie \
    --output_dir /disc2/tianbaili/transformers/classifier_outputs/ernie_fine_tune \
    --do_train \
    --eval_all_checkpoints \
