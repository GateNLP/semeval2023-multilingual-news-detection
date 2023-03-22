#!/bin/bash
TOKENIZERS_PARALLELISM=false python train.py \
        --project_name semeval \
        --experiment_name multilang \
        --train_languages en it ru ge fr po \
        --val_languages en it ru ge fr po \
        --pretrained_name bert-base-multilingual-cased \
        --epochs 20 \
        --max_seq_len 256 \
        --batch_size 16 \
        --warmup_ratio 0.2 \
        --weight_decay 0.1 \
        --seed 10 \
        --with_class_weights \
        --with_unlabelled \
        --monitor_metric loss \
        --offline

TOKENIZERS_PARALLELISM=false python train.py \
        --project_name semeval \
        --experiment_name en_monolang \
        --train_languages en \
        --val_languages en \
        --pretrained_name roberta-base \
        --epochs 20 \
        --max_seq_len 256 \
        --batch_size 32 \
        --warmup_ratio 0.2 \
        --weight_decay 0.1 \
        --seed 10 \
        --with_class_weights \
        --with_unlabelled \
        --monitor_metric loss \
        --offline