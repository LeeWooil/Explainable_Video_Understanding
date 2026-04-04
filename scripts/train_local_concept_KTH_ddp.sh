#!/usr/bin/bash
#SBATCH -J KTH_train_local_concept_21_block11
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH -p batch_vll
#SBATCH -w vll2
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.err

source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start

PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/train_local_concept.py"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    $PYTHON_SCRIPT \
    --anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH/train.csv \
    --val-anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH/val.csv \
    --data-root /local_datasets/kth/video \
    --pseudo-mask-root /data/dataset/VideoXAI/optical_flow/kth_flow_21 \
    --backbone vmae_vit_base_patch16_224 \
    --finetune /data/lwi2765/repos/VideoMAE/videoMAE/result/KTH/OUT/KTH_videomae_finetune.pth \
    --data-set kth \
    --nb-classes 6 \
    --num-concepts 21 \
    --batch-size 16 \
    --epochs 30 \
    --lr 1e-2 \
    --num-workers 8 \
    --device cuda \
    --block-index 11 \
    --num-frames 16 \
    --num-segments 1 \
    --sampling-rate 4 \
    --tubelet-size 2 \
    --input-size 224 \
    --patch-size 16 \
    --eval-threshold 0.1 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 0.0 \
    --view-mode center_uniform \
    --output-dir /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/block_ablation \
    --save-preview-every 5 \
    --preview-max-samples 4
slack_end $?
    # --use-pos-weight \
    # --pos-weight-max 10.0 \