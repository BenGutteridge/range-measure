#!/bin/bash

# All commands for running synthetic tasks

# ------- Compute task ranges for *node-level* synthetic tasks (Figure 2) ------------------------------------------------------------
python main.py --dataset sbm --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset sbmsparse --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset erdos --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset erdossparse --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset ba --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset linegraphs --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2
python main.py --dataset gridgraphs --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 2


# ------- Compute task ranges for *graph-level* synthetic tasks, using *Hessian* -------------------------------------------
python main.py --dataset sbm_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset sbmsparse_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset erdos_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset erdossparse_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset ba_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset linegraphs_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN
python main.py --dataset gridgraphs_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN


# ------- Compute task ranges for *graph-level* synthetic tasks (as above), instead using *final layer Jacobian* ------------
python main.py --dataset sbm_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN  --use_jacobian
python main.py --dataset sbmsparse_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian
python main.py --dataset erdos_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian
python main.py --dataset erdossparse_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian
python main.py --dataset ba_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian
python main.py --dataset linegraphs_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian
python main.py --dataset gridgraphs_graph --skip --num_layers 1 --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 2 --pool MEAN --use_jacobian


# ------- Node-level linegraph exp for every depth of GCN and 4 seeds, for power task with k=5 (Figure 4)---------------------------
depth_ls=$(seq 1 8)
seed_ls=$(seq 1 4)
for i in $depth_ls; do
  for seed in $seed_ls; do
    python main.py --dataset linegraphs --skip --num_layers $i --seed $seed --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn opp_lin --track_range --max_epochs 20
  done
done


# ------- Graph-level linegraph exp for every depth of GCN and 4 seeds, for power task with k=5 (Figure 5)---------------------------
# Both for Hessian and for final layer Jacobian
depth_ls=$(seq 1 8)
seed_ls=$(seq 1 4)
for i in $depth_ls; do
  for seed in $seed_ls; do
    python main.py --dataset linegraphs_graph --skip --num_layers $i --seed $seed --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 50 --pool MEAN
    python main.py --dataset linegraphs_graph --skip --num_layers $i --seed $seed --lr_exp --distance_fn adjacency_self_loop_power_sym_5 --interaction_fn L2_norm --track_range --max_epochs 50 --pool MEAN --use_jacobian
  done
done


# ------- Node-level Power, Dirac/hop and Rectangle/nbhd tasks on gridgraphs for a 5-layer GCN (Figure 3)---------------------------
distance_ls=("adjacency_self_loop_power_sym_" "hop_" "nbhd_")
k_ls=$(seq 1 8)
seed_ls=$(seq 1 4)
for distance in $distance_ls; do
  for k in $k_ls; do
    for seed in $seed_ls; do
      distance_fn="${distance}_${k}"  # Concatenate integer to string
      python main.py --dataset gridgraphs --skip --num_layers 5 --seed $seed --lr_exp --distance_fn $distance_fn --interaction_fn opp_lin --track_range --max_epochs 20
    done
  done
done


# ------- Graph-level Power, Dirac/hop and Rectangle/nbhd tasks on gridgraphs for a 5-layer GCN ---------------------------
# Both for Hessian and for final layer Jacobian
distance_ls=("adjacency_self_loop_power_sym_" "hop_" "nbhd_")
k_ls=$(seq 1 8)
seed_ls=$(seq 1 4)
for distance in $distance_ls; do
  for k in $k_ls; do
    for seed in $seed_ls; do
      distance_fn="${distance}_${k}"  # Concatenate integer to string
      python main.py --dataset gridgraphs_graph --skip --num_layers 5 --seed $seed --lr_exp --distance_fn $distance_fn --interaction_fn opp_lin --track_range --max_epochs 20 --pool MEAN
      python main.py --dataset gridgraphs_graph --skip --num_layers 5 --seed $seed --lr_exp --distance_fn $distance_fn --interaction_fn opp_lin --track_range --max_epochs 20 --pool MEAN --use_jacobian
    done
  done
done
