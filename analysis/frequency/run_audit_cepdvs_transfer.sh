#!/usr/bin/env bash
set -euo pipefail

dataset_root="${1:-/home/user/Datasets/kpm-own/CEP-DVS}"
reference_dir="${2:-/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/smoke_20261014T072630Z}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../.." && pwd -P)"
code_root="$(realpath /home/user/kpm/kpm)"
data_root="$(realpath /home/user/Datasets/kpm-own)"
dataset_root="$(realpath "$dataset_root")"
reference_dir="$(realpath "$reference_dir")"

case "$project_root/" in
  "$code_root/"*) ;;
  *) echo "Project resolves outside $code_root: $project_root" >&2; exit 1 ;;
esac
case "$dataset_root/" in
  "$data_root/"*) ;;
  *) echo "Dataset resolves outside $data_root: $dataset_root" >&2; exit 1 ;;
esac
case "$reference_dir/" in
  "$code_root/"*) ;;
  *) echo "Reference resolves outside $code_root: $reference_dir" >&2; exit 1 ;;
esac

for input_path in "$dataset_root/data/edge" "$dataset_root/data/dvs_processed" "$dataset_root/pathFile.csv"; do
  resolved="$(realpath "$input_path")"
  case "$resolved/" in
    "$data_root/"*) ;;
    *) echo "Input resolves outside $data_root: $resolved" >&2; exit 1 ;;
  esac
done
for input_path in "$reference_dir/run.json" "$reference_dir/per_sample_channel.csv"; do
  [[ -f "$input_path" ]] || { echo "Reference file missing: $input_path" >&2; exit 1; }
done

results_parent="/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs"
resolved_results="$(realpath -m "$results_parent")"
case "$resolved_results/" in
  "$code_root/"*) ;;
  *) echo "Results resolve outside $code_root: $resolved_results" >&2; exit 1 ;;
esac

run_id="edge_pair_audit_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$resolved_results/$run_id"
[[ ! -e "$output_dir" ]] || { echo "Output already exists: $output_dir" >&2; exit 1; }

echo "Project: $project_root"
echo "Dataset: $dataset_root"
echo "Reference: $reference_dir"
echo "Output: $output_dir"
cd "$project_root"
python -m analysis.frequency.audit_cepdvs_transfer \
  --dataset-root "$dataset_root" \
  --reference-run "$reference_dir/run.json" \
  --output-dir "$output_dir"
