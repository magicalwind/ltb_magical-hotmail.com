#! /usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export SQUAD_DIR=/disc2/tianbaili/cmrc2018/squad-style-data
export OUTPUT_DIR=/disc2/tianbaili/transformers/outputs/squad

python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path /disc2/tianbaili/transformers/outputs/roberta_normal_2400 \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/cmrc2018_train.json \
    --predict_file $SQUAD_DIR/real_sample2.json \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR/ \
    --per_gpu_eval_batch_size=3 \
    --per_gpu_train_batch_size=3 \
    --max_answer_length 100 \
    --do_real_pred \
    --questions_json $SQUAD_DIR/question_sample.json \
    --documents_json $SQUAD_DIR/document.json \
#    --do_train \
#    --eval_all_checkpoints \
#    --evaluate_during_training \
#    --tokenizer_name hfl/chinese-roberta-wwm-ext-large \
#    --config_name /disc2/tianbaili/transformers/config_xlnet.json \
#    --version_2_with_negative

