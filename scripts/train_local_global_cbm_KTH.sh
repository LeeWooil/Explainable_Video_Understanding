#!/usr/bin/bash
#SBATCH -J KTH_train_local_global_cbm_local_only__no_bias_mean_logit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_vll
#SBATCH -w vll3
#SBATCH -t 4-00:00:00
#SBATCH -o log/%A-%x.out
#SBATCH -e log/%A-%x.err

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_SCRIPT="${REPO_ROOT}/train_local_global_cbm.py"

python \
    $PYTHON_SCRIPT \
    --anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH/train.csv \
    --val-anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH/val.csv \
    --data-root /local_datasets/kth/video \
    --backbone vmae_vit_base_patch16_224 \
    --finetune /data/lwi2765/repos/VideoMAE/videoMAE/result/KTH/OUT/KTH_videomae_finetune.pth \
    --data-set kth \
    --nb-classes 6 \
    --num-concepts 21 \
    --batch-size 16 \
    --num-workers 4 \
    --device cuda \
    --block-index 6 \
    --num-frames 16 \
    --num-segments 1 \
    --sampling-rate 4 \
    --tubelet-size 2 \
    --input-size 224 \
    --patch-size 16 \
    --localizer-ckpt ${REPO_ROOT}/runs/local_concept/no_bias/KTH_21_no_bias/best.pt \
    --pooling mean \
    --pool-source logit \
    --fusion-mode local \
    --global-label-dir /data/lwi2765/repos/XAI/Video_Language_XAI/Concept_exraction/Trajectory_based_method/result/KTH_global_0.95/global_labels_vis_21concepts_03-19_21-47-01 \
    --video-anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH \
    --save-dir ${REPO_ROOT}/runs/CBM_result/KTH/local_only \
    --use-mlp
