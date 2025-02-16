#!/bin/bash

# # WARNING: may produce OOM errors if running without a GPU

# Test node-level synthetic experiments
python main.py \
--dataset linegraphs \
--model_type GCN \
--skip \
--num_layers 3 \
--seed 0 \
--lr_exp \
--distance_fn adjacency_self_loop_power_sym_5 \
--interaction_fn opp_lin \
--track_range \
--track_epoch 1 \
--max_epochs 2 \
--num_graphs 50

# Test graph-level synthetic experiments
python main.py \
--dataset linegraphs_graph \
--model_type GCN \
--skip \
--hidden_dim 64 \
--num_layers 3 \
--seed 0 \
--lr_exp \
--distance_fn adjacency_self_loop_power_sym_5 \
--interaction_fn L2_norm \
--track_range \
--track_epoch 1 \
--max_epochs 1 \
--pool MEAN
--num_graphs 50


# Test LRGB experiments 
cd lrgb_exps

# 1. Train a small model from scratch
python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml \
    out_dir results/debug \
    optim.max_epoch 2 \
    train.auto_resume False \
    longrange.track_range_measure True \
    posenc_RWSE.enable False \
    posenc_LapPE.enable False \
    dataset.node_encoder_name AtomDiff

# 2. Evaluate Jacobians on the small model
python main.py --cfg configs/LRGB-tuned/peptides-func-GCN.yaml \
    out_dir results/debug \
    train.auto_resume True \
    train.batch_size 1 \
    train.mode eval_range \
    longrange.subset 5 \
    longrange.track_range_measure True \
    longrange.epoch 1 \
    longrange.split val \
    posenc_RWSE.enable False \
    posenc_LapPE.enable False \
    dataset.node_encoder_name AtomDiff \
    train.ckpt_clean False
