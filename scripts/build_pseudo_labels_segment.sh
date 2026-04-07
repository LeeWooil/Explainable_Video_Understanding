#!/usr/bin/bash
#SBATCH -J ssv2_chiral_pseudo_label_154
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=150G
#SBATCH -p batch_vll
#SBATCH -w vll5
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.err

source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/build_pseudo_labels.py"

# 실행 명령어
python  \
    $PYTHON_SCRIPT \
    --trajectory-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_trajectory_L16 \
    --raw-flow-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_flow \
    --output-root /data/dataset/VideoXAI/pseudo_mask/ssv2_chirality/threshold_0.90_154concepts \
    --global-result-dir /data/dataset/VideoXAI/trajectory_per_sample/SSv2_chirality/global_0.95/global_partition_partition8_154clusters_03-24_17-45-14 \
    --num-workers ${SLURM_CPUS_PER_GPU:-8} \
    --patch-size 16 \
    --tubelet-size 2 \
    --anno-path /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/train.csv \
               /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/val.csv \
    --data-root /local_datasets/something-something-v2 \
    --data-set SSv2_chiral \
    --num-frames 16 \
    --input-size 224 \
    --short-side-size 224
slack_end $?