#!/bin/bash
# head /home/data/SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips.csv
# head /home/data/SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv
# head /home/data/SEP-28k/ml-stuttering-events-dataset/fluencybank_labels.csv
#Select cuda devices to be visible by setting env variable:
# CUDA_DEVICE_ORDER
# options: FASTEST_FIRST, PCI_BUS_ID, (default is FASTEST_FIRST)
export CUDA_DEVICE_ORDER=PCI_BUS_ID # setting order to match the PCI BUS Order, so it matches nvidia-smi output
export CUDA_VISIBLE_DEVICES=4,5,6


python -m mlstutterdetection fine-tune-multilabel-w2v2-classifier \
        --sep28k-labels /home/data/SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_clips.csv  \
        --sep28k-episodes /home/data/SEP-28k/ml-stuttering-events-dataset/SEP-28k-Extended_episodes.csv  \
        --fluencybank-labels /home/data/SEP-28k/ml-stuttering-events-dataset/fluencybank_labels.csv  \
        --audio-dir /home/data/SEP-28k/ml-stuttering-events-dataset  \
        --ksof-labels /home/data/KST/segments/kassel-state-of-fluency_partitioned.csv  \
        --ksof-audio-dir /home/data/KST  \
        --train-data mixed \
        --clf-type mean \
        --focal-loss \
        --focal-loss-alpha 0.7 \
        --focal-loss-gamma 3 \
        --aux-col language  \
        --main-loss-weight 0.99  \
        --freeze-encoder  \
        --eval-accumulation-steps 10 \
        --batch-size 24  \
        --gradient-cum-steps 8 \
        --model /home/data/stuttering_models/sep28k_w2v2_mtl_multilabel_mixed/2024_8_21_7_41_41_multiclass_language_0.992_24_mixed_finetuned \
        --epochs 5 \
        --early-stopping 5 \
        --num-hidden-layers 4 \
        --log-dir /home/haas/test_logs
