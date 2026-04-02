#!/usr/bin/bash
#SBATCH -J SSv2_precompute_target_cache_154
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_vll
#SBATCH -w vll3
#SBATCH -t 4-00:00:00
#SBATCH -o log/%A-%x.out
#SBATCH -e log/%A-%x.err

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_SCRIPT="${REPO_ROOT}/train_local_concept.py"
TARGET_CACHE_ROOT="${REPO_ROOT}/cache/local_concept_targets_ssv2_chiral"

python \
    $PYTHON_SCRIPT \
    --anno-path /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/train.csv \
    --val-anno-path /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/val.csv \
    --data-root /local_datasets/something-something-v2-mp4 \
    --pseudo-mask-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_flow_154 \
    --backbone vmae_vit_base_patch16_224 \
    --finetune /data/lwi2765/repos/VideoMAE/videoMAE/videomae_weight/ssv2_finetune_800.pth \
    --data-set SSv2_chiral \
    --nb-classes 32 \
    --batch-size 8 \
    --epochs 20 \
    --lr 5e-3 \
    --num-workers 8 \
    --device cuda \
    --block-index 6 \
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
    --target-cache-root ${TARGET_CACHE_ROOT} \
    --precompute-target-cache-only
