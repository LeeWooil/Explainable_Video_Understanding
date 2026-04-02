#!/usr/bin/bash
#SBATCH -J SSv2_chiral_pseudo_label_154
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=150G
#SBATCH -p batch_vll
#SBATCH -w vll5
#SBATCH -t 4-00:00:00
#SBATCH -o log/%A-%x.out
#SBATCH -e log/%A-%x.err

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_SCRIPT="${REPO_ROOT}/scripts/build_pseudo_labels.py"

# 실행 명령어
python  \
    $PYTHON_SCRIPT \
    --trajectory-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_trajectory_L16 \
    --raw-flow-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_flow \
    --output-root /data/dataset/VideoXAI/optical_flow/ssv2_chiral_flow_154 \
    --global-result-dir /data/dataset/VideoXAI/trajectory_per_sample/SSv2_chirality/global_0.95/global_partition_partition8_154clusters_03-24_17-45-14 \
    --trajectory-start-mode segment \
    --num-workers ${SLURM_CPUS_PER_GPU:-8}
