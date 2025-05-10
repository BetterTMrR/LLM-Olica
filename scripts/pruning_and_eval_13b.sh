#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python model_pruning.py \
--base_model 13b \
--datasets bookcorpus+alpaca \
--num_samples 128 \
--seqlen 128 \
--percdamp 0.5 \
--mlp_num 6 \
--sparsity 0.2 \
--ratio 0.03 \
--k 3 \
--save_dir ./pruned_models

CUDA_VISIBLE_DEVICES=2 python model_evaluate.py \
--base_model 13b \
--sparsity 0.2 \
--tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
--save_dir ./pruned_models