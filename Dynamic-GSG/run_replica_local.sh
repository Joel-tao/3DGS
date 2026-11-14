#!/usr/bin/env bash
set -euo pipefail
cd /home/joe/Dynamic-GSG
if ! mountpoint -q /media/joe/Data; then
  echo "Data 盘未挂载。请先运行：udisksctl mount -b /dev/disk/by-uuid/F03AD16D3AD130F8" >&2
  exit 1
fi
export CUDA_HOME=/home/joe/anaconda3/envs/dgsg
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME=/media/joe/Data/Dynamic-GSG/cache/huggingface
export TORCH_HOME=/media/joe/Data/Dynamic-GSG/cache/torch
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
exec "$CUDA_HOME/bin/python" -u scripts/dynamic_gsg_real_ssim.py configs/replica/dgsg_local.py "$@"
