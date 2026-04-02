#!/usr/bin/bash
#SBATCH -J KTH_train_local_concept_21_pos_weight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH -p batch_vll
#SBATCH -w vll3
#SBATCH -t 4-00:00:00
#SBATCH -o log/%A-%x.out
#SBATCH -e log/%A-%x.err

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_SCRIPT="${REPO_ROOT}/train_local_concept.py"

python  \
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
    --batch-size 32 \
    --epochs 15 \
    --lr 5e-3 \
    --num-workers 4 \
    --device cuda \
    --block-index 6 \
    --num-frames 16 \
    --num-segments 1 \
    --sampling-rate 4 \
    --tubelet-size 2 \
    --input-size 224 \
    --patch-size 16 \
    --eval-threshold 0.2 \
    --early-stopping-patience 5 \
    --early-stopping-min-delta 0.0 \
    --use-pos-weight \
    --pos-weight-max 10.0 \
    --output-dir ${REPO_ROOT}/runs/KTH_21_pos_weight \
    --save-preview-every 5 \
    --preview-max-samples 4 \
    --num-concepts 21
    
