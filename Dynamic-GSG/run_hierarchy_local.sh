#!/usr/bin/env bash
set -euo pipefail
cd /home/joe/Dynamic-GSG
export PYTHONDONTWRITEBYTECODE=1
hierarchy_python=/home/joe/anaconda3/envs/dgsg/bin/python
hierarchy_results=${1:-/media/joe/Data/Dynamic-GSG/experiments/room0_local200}
"$hierarchy_python" scripts/build_hierarchy_local.py "$hierarchy_results"
"$hierarchy_python" scripts/plot_hierarchy_local.py "$hierarchy_results"
"$hierarchy_python" scripts/query_hierarchy_local.py "$hierarchy_results" --category pillow --parent couch --output "$hierarchy_results/query_pillows_on_couch_local.json"
"$hierarchy_python" scripts/render_preview_local.py "$hierarchy_results" --frames 0 140 --query-result "$hierarchy_results/query_pillows_on_couch_local.json" --output "$hierarchy_results/query_pillows_on_couch_local_preview.png"
