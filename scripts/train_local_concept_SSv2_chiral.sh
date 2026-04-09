#!/usr/bin/bash
#SBATCH -J SSv2_train_local_concept_154
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=40G
#SBATCH -p batch_vll
#SBATCH -w vll6
#SBATCH -t 4-00:00:00
#SBATCH -o /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.out
#SBATCH -e /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/log/%A-%x.err

source /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/scripts/slack_notify.sh
slack_start


source /data/lwi2765/anaconda3/etc/profile.d/conda.sh
conda activate DANCE

echo "Using python: $(which python)"
python --version
echo "Using torchrun: $(which torchrun)"
echo "Using pip: $(which pip)"
python -m pip show tqdm | sed -n '1,20p'

PYTHON_SCRIPT="/data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/train_local_concept.py"
NUM_GPUS=${SLURM_GPUS_ON_NODE:-8}
TARGET_CACHE_ROOT="/data/dataset/VideoXAI/pseudo_mask/ssv2_chirality/cache"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    $PYTHON_SCRIPT \
    --anno-path /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/train.csv \
    --val-anno-path /data/lwi2765/repos/XAI/PCBEAR/dataset/ssv2_chiral/val.csv \
    --data-root /local_datasets/something-something-v2-mp4 \
    --pseudo-mask-root /data/dataset/VideoXAI/pseudo_mask/ssv2_chirality/threshold_0.90_154concepts \
    --backbone vmae_vit_base_patch16_224 \
    --finetune /data/lwi2765/repos/VideoMAE/videoMAE/videomae_weight/ssv2_finetune_800.pth \
    --data-set SSv2_chiral \
    --nb-classes 32 \
    --batch-size 32 \
    --epochs 20 \
    --lr 5e-3 \
    --num-workers 2 \
    --device cuda \
    --block-index 11 \
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
    --output-dir /data/lwi2765/repos/XAI/Video_Language_XAI/CBM_training_ver2/runs/SSv2_chiral/SSv2_block_11 \
    --save-preview-every 5 \
    --preview-max-samples 4 \
    --predownsampled
slack_end $?
# If you want cache generation only:
#    replace `--precompute-target-cache` with `--precompute-target-cache-only`
#
# If you want to reuse an existing cache without rebuilding it:
#    remove `--precompute-target-cache`
