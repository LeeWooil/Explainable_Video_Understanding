#!/usr/bin/bash
#SBATCH -J KTH_block_ablation
#SBATCH -w vll1
#SBATCH --array=0-3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_vll
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A_%a-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A_%a-%x.err

# Usage: sbatch ablation_block_array_KTH_ddp.sh
# 한 번의 sbatch로 4개 블록(3, 6, 9, 11)을 동시에 각각 별도 GPU에서 실행
# 로그: %A = job ID, %a = array task index (0-3)
source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start
# array index → block index 매핑
#   0 → 3  (초기 블록: low-level spatial feature)
#   1 → 6  (중간 블록: 기존 baseline과 동일)
#   2 → 9  (후반 블록: high-level semantic feature)
#   3 → 11 (최종 블록: classification에 가장 가까운 feature)
BLOCKS=(0 5 7 10)
BLOCK=${BLOCKS[$SLURM_ARRAY_TASK_ID]}

PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/train_local_concept.py"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
TIMESTAMP=$(date +"%m-%d_%H-%M-%S")

echo "===== [array_task=${SLURM_ARRAY_TASK_ID}] Training block_index=${BLOCK} on ${NUM_GPUS} GPUs ====="
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
    --lr 5e-3 \
    --num-workers 4 \
    --device cuda \
    --block-index ${BLOCK} \
    --num-frames 16 \
    --num-segments 1 \
    --sampling-rate 4 \
    --tubelet-size 2 \
    --input-size 224 \
    --patch-size 16 \
    --eval-threshold 0.2 \
    --view-mode center_uniform\
    --early-stopping-patience 5 \
    --early-stopping-min-delta 0.0 \
    --output-dir /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/block_ablation/KTH_21_block${BLOCK}_${TIMESTAMP} \
    --save-preview-every 5 \
    --preview-max-samples 10
echo "===== [array_task=${SLURM_ARRAY_TASK_ID}] Done block_index=${BLOCK} ====="

slack_end $?
