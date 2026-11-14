#!/usr/bin/env bash
set -euo pipefail
cd /home/joe/Dynamic-GSG
if ! mountpoint -q /media/joe/Data; then
  echo '请先挂载 Data 盘。' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
hierarchy_python=/home/joe/anaconda3/envs/dgsg/bin/python
hierarchy_results=${1:-/media/joe/Data/Dynamic-GSG/experiments/room0_local200}
"$hierarchy_python" scripts/classify_objects_model.py "$hierarchy_results"
"$hierarchy_python" scripts/build_hierarchy_local.py "$hierarchy_results" --model-classifications "$hierarchy_results/model_classifications.json" --output-stem scene_graph_hierarchy_model
"$hierarchy_python" scripts/plot_hierarchy_local.py "$hierarchy_results" --graph-stem scene_graph_hierarchy_model
"$hierarchy_python" scripts/query_hierarchy_local.py "$hierarchy_results" --graph-stem scene_graph_hierarchy_model --category pillow --parent couch --output "$hierarchy_results/query_pillows_on_couch_model.json"
"$hierarchy_python" scripts/render_preview_local.py "$hierarchy_results" --frames 0 140 --query-result "$hierarchy_results/query_pillows_on_couch_model.json" --output "$hierarchy_results/query_pillows_on_couch_model_preview.png"
"$hierarchy_python" scripts/compare_hierarchy_models.py "$hierarchy_results"
