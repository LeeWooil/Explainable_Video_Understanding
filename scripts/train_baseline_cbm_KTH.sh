#!/usr/bin/bash
#SBATCH -J KTH_baseline_cbm_aligned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G
#SBATCH -p batch_vll
#SBATCH -w vll1
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.err

source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start

PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/train_baseline_cbm.py"

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
    --block-index 11 \
    --num-frames 16 \
    --num-segments 1 \
    --sampling-rate 4 \
    --tubelet-size 2 \
    --input-size 224 \
    --patch-size 16 \
    --global-label-dir /data/lwi2765/repos/XAI/Video_Language_XAI/Concept_exraction/Trajectory_based_method/result/KTH_global_0.95/global_labels_vis_21concepts_03-19_21-47-01 \
    --video-anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH \
    --save-dir /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/CBM_result/KTH/baseline_cbm \
    --use-mlp \
    --loss-mode concept \
    --deterministic-spatial \
    --proj-steps 3000 \
    --n-iters 30000

slack_end $?
