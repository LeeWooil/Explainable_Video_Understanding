#!/usr/bin/bash
#SBATCH -J KTH_local_global_cbm_flatten
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G
#SBATCH -p batch_vll
#SBATCH -w vll3
#SBATCH -t 4-00:00:00
#SBATCH -o /data/junpyohong/project/Explainable_Video_Understanding/log/%A-%x.out
#SBATCH -e /data/junpyohong/project/Explainable_Video_Understanding/log/%A-%x.err

source /data/junpyohong/project/dance_of/scripts/slack_notify.sh
slack_start

source /data/junpyohong/anaconda3/etc/profile.d/conda.sh
conda activate DANCE

PYTHON_SCRIPT="/data/junpyohong/project/Explainable_Video_Understanding/train_local_global_cbm.py"

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
    --localizer-ckpt /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/block_ablation_aligned/block11/best.pt \
    --pooling flatten \
    --pool-source logit \
    --fusion-mode local \
    --fusion-gate 0.3 \
    --global-pose-dir /data/lwi2765/repos/XAI/PCBEAR/CBM_training/Final_result/KTH/Trajectory/global_threshold0.95/kth_global_labels_vis_21concepts_03-19_21-47-01_03-19_22-11-25/pose \
    --global-backbone-train /data/lwi2765/repos/XAI/PCBEAR/CBM_training/results/Features/KTH/kth_train_vmae_vit_base_patch16_224.pt \
    --global-backbone-val /data/lwi2765/repos/XAI/PCBEAR/CBM_training/results/Features/KTH/kth_val_vmae_vit_base_patch16_224.pt \
    --global-label-dir /data/lwi2765/repos/XAI/Video_Language_XAI/Concept_exraction/Trajectory_based_method/result/KTH_global_0.95/global_labels_vis_21concepts_03-19_21-47-01 \
    --video-anno-path /data/lwi2765/repos/XAI/Video_Language_XAI/dataset/KTH \
    --save-dir /data/junpyohong/project/Explainable_Video_Understanding/runs/CBM_result/KTH/local_global_cbm_flatten \
    --use-mlp \
    --deterministic-spatial
slack_end $?
