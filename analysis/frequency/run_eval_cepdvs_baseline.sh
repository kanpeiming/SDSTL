#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../.." && pwd -P)"
code_root="$(realpath /home/user/kpm/kpm)"
data_root="$(realpath /home/user/Datasets/kpm-own)"
dataset_root="$(realpath /home/user/Datasets/kpm-own/CEP-DVS)"
checkpoint="$(realpath /home/user/kpm/kpm/results/SDSTL/baseline/checkpoints_CEP-DVS/SDSTL_CEP-DVS_TET_img48_T10_seed1000_dvs1_enc-TET_lr0.001_VGGSNN_namebaseline.pth)"
results_parent="$(realpath -m /home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline)"

case "$project_root/" in "$code_root/"*) ;; *) echo "Project outside code root" >&2; exit 1 ;; esac
case "$dataset_root/" in "$data_root/"*) ;; *) echo "Dataset outside data root" >&2; exit 1 ;; esac
case "$checkpoint/" in "$code_root/"*) ;; *) echo "Checkpoint outside code root" >&2; exit 1 ;; esac
case "$results_parent/" in "$code_root/"*) ;; *) echo "Results outside code root" >&2; exit 1 ;; esac

run_id="eval_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$results_parent/$run_id"
[[ ! -e "$output_dir" ]] || { echo "Output exists: $output_dir" >&2; exit 1; }
mkdir -p "$output_dir"
cd "$project_root"

python -m analysis.frequency.eval_cepdvs_baseline \
  --dataset-root "$dataset_root" \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir" \
  --batch-size 8 2>&1 | tee "$output_dir/console.log"

echo "Results: $output_dir"
