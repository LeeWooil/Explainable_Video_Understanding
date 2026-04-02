#!/usr/bin/bash
#SBATCH -J KTH_chiral_pseudo_label_filtering_slidingwindow
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
PYTHON_SCRIPT="${REPO_ROOT}/scripts/build_pseudo_labels.py"

# 실행 명령어
python  \
    $PYTHON_SCRIPT \
    --flow-root /data/dataset/VideoXAI/optical_flow/kth_flow_raw \
    --output-root /data/dataset/VideoXAI/optical_flow/kth_flow_sliding_window_filtering \
    --trajectory-length 16\
    --trajectory-stride 4\
    --motion-threshold-percentile 95 \
    --saliency-mask-root /data/dataset/VideoXAI/saliency_mask/kth \
    --saliency-filter-mode start_frame \
