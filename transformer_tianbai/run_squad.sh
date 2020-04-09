#! /usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SQUAD_DIR=/disc2/tianbaili/cmrc2018/squad-style-data
export OUTPUT_DIR=/disc2/tianbaili/transformers/outputs/squad

python -m torch.distributed.launch --nproc_per_node=4 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/complete_train.json \
    --predict_file $SQUAD_DIR/real_sample2.json \
    --learning_rate 3e-5 \
    --num_train_epochs 18 \
    --max_seq_length 384 \
    --n_best_size 10 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR/ \
    --per_gpu_eval_batch_size=3 \
    --per_gpu_train_batch_size=3 \
    --tokenizer_name hfl/chinese-roberta-wwm-ext-large \
    --eval_all_checkpoints \
    --evaluate_during_training \
#    --config_name /disc2/tianbaili/transformers/config_xlnet.json \
#    --version_2_with_negative

