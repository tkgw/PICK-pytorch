#!/bin/sh

# env CUDA_VISIBLE_DEVICES=0 \
#     python3 -m torch.distributed.launch \
#     --nnode=1 \
#     --node_rank=0 \
#     --nproc_per_node=1 \
#     train.py \
#     --config config.json --device 0 --local_world_size 1


env CUDA_VISIBLE_DEVICES=0 \
    python3 train.py \
    --config config.json --distributed false --device 0 --local_world_size 1
