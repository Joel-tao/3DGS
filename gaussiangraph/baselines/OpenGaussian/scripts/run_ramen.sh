#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_root="${OPENGS_WORK_ROOT:-/media/joe/Data/gaussiangraph}"
python_bin="${OPENGS_PYTHON:-$work_root/envs/opengs/bin/python}"
scene_dir="${RAMEN_DATA:-$work_root/data/lerf_ovs/ramen}"
model_dir="${RAMEN_MODEL:-$work_root/outputs/ramen_baseline}"
gt_dir="${RAMEN_GT:-$work_root/data/lerf_ovs/label/ramen/gt}"
cd "$repo_dir"
case "${1:-check}" in
  check)
    "$python_bin" scripts/check_ramen.py --data "$scene_dir" --gt "$gt_dir"
    ;;
  train)
    "$python_bin" scripts/check_ramen.py --data "$scene_dir" --gt "$gt_dir" --training-only
    mkdir -p "$model_dir"
    git rev-parse HEAD > "$model_dir/upstream_commit.txt"
    git diff > "$model_dir/local_changes.patch"
    "$python_bin" -m pip freeze > "$model_dir/environment.txt"
    # Keep the official ramen schedule and clustering parameters. CPU storage and
    # save_memory reduce GPU residency without changing image resolution.
    command=("$python_bin" -u train.py -s "$scene_dir" -m "$model_dir" \
      --port 6010 --iterations 70000 --start_ins_feat_iter 30000 \
      --start_root_cb_iter 40000 --start_leaf_cb_iter 50000 \
      --sam_level 3 --root_node_num 64 --leaf_node_num 10 \
      --pos_weight 0.5 --loss_weight 0.01 --save_memory --data_device cpu \
      --test_iterations 30000 --eval "${@:2}")
    "$python_bin" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:],indent=2))' \
      "$model_dir/command.json" "${command[@]}"
    if [[ -f "$work_root/data/lerf_ovs/feature_source.json" ]]; then
      cp "$work_root/data/lerf_ovs/feature_source.json" "$model_dir/feature_source.json"
    fi
    exec "${command[@]}"
    ;;
  render)
    exec "$python_bin" render_lerf_by_text.py -m "$model_dir" --scene_name ramen "${@:2}"
    ;;
  eval)
    exec "$python_bin" scripts/compute_lerf_iou.py --scene_name ramen \
      --gt "$gt_dir" --pred "$model_dir/text2obj/ours_70000/renders_cluster_silhouette" \
      --output "$model_dir/metrics.json" "${@:2}"
    ;;
  *) echo 'Usage: bash scripts/run_ramen.sh {check|train|render|eval} [extra arguments]' >&2; exit 2 ;;
esac
