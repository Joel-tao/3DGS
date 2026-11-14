#!/usr/bin/env bash
set -euo pipefail
cd /home/joe/Dynamic-GSG
if ! mountpoint -q /media/joe/Data; then
  echo "Data 盘未挂载。请先运行：udisksctl mount -b /dev/disk/by-uuid/F03AD16D3AD130F8" >&2
  exit 1
fi
export HF_HOME=/media/joe/Data/Dynamic-GSG/cache/huggingface
export TORCH_HOME=/media/joe/Data/Dynamic-GSG/cache/torch
export PYTHONDONTWRITEBYTECODE=1
exec /home/joe/anaconda3/envs/dgsg/bin/python viz_scripts/classifier_vis.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200/config_replay.py
