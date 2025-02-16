#!/bin/bash

cd lrgb_exps

paths=(
    "configs/LRGB-tuned/peptides-func-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-func-GCN.yaml"
    "configs/LRGB-tuned/peptides-func-GINE.yaml"
    "configs/LRGB-tuned/peptides-func-GPS.yaml"
    "configs/LRGB-tuned/peptides-struct-GatedGCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GCN.yaml"
    "configs/LRGB-tuned/peptides-struct-GINE.yaml"
    "configs/LRGB-tuned/peptides-struct-GPS.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GatedGCN.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GCN.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GINE.yaml"
    "configs/LRGB-tuned/vocsuperpixels-GPS.yaml"
)

for i in "${!paths[@]}"; do
    echo "Running job $i"
    python main.py --cfg "${paths[$i]}" longrange.track_range_measure train.ckpt_period 1 train.ckpt_clean False out_dir results/sota seed 0
done