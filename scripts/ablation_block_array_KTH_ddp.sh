#!/usr/bin/bash
#SBATCH -J KTH_block_ablation_new
#SBATCH -w vll6
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_vll
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.err

# Usage: sbatch ablation_block_array_KTH_ddp.sh
# 한 번의 sbatch로 8개 GPU를 사용해 block ablation을 순차 실행
# 앞 블록 실험이 끝나면 다음 블록 실험이 이어서 실행됨
source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start
BLOCKS=(11 0 3 5 6 7 9 10)

PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/train_local_concept.py"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-8}
JOB_TIMESTAMP=$(date +"%m-%d_%H-%M-%S")
EXIT_CODE=0

for BLOCK in "${BLOCKS[@]}"; do
    RUN_TIMESTAMP=$(date +"%m-%d_%H-%M-%S")
    echo "===== Training block_index=${BLOCK} on ${NUM_GPUS} GPUs ====="

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
        --view-mode center_uniform \
        --early-stopping-patience 5 \
        --early-stopping-min-delta 0.0 \
        --output-dir /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/block_ablation_new/KTH_21_job${JOB_TIMESTAMP}_block${BLOCK}_${RUN_TIMESTAMP} \
        --save-preview-every 5 \
        --preview-max-samples 10 \
        --deterministic-spatial

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "===== Failed block_index=${BLOCK} with exit code ${EXIT_CODE}. Stop remaining runs. ====="
        break
    fi

    echo "===== Done block_index=${BLOCK} ====="
done

slack_end ${EXIT_CODE}
exit ${EXIT_CODE}
