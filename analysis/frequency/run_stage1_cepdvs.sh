#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
dataset_root="${2:-/home/user/Datasets/kpm-own/CEP-DVS}"
case "$mode" in
  smoke) per_class=1; plot_count=1 ;;
  full) per_class=5; plot_count=3 ;;
  *) echo "Usage: bash analysis/frequency/run_stage1_cepdvs.sh [smoke|full] [CEP-DVS root]" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../.." && pwd -P)"
code_root="$(realpath /home/user/kpm/kpm)"
data_root="$(realpath /home/user/Datasets/kpm-own)"
dataset_root="$(realpath "$dataset_root")"

case "$project_root/" in
  "$code_root/"*) ;;
  *) echo "Project resolves outside $code_root: $project_root" >&2; exit 1 ;;
esac
case "$dataset_root/" in
  "$data_root/"*) ;;
  *) echo "Dataset resolves outside $data_root: $dataset_root" >&2; exit 1 ;;
esac

rgb_dir="$(realpath "$dataset_root/data/img")"
dvs_dir="$(realpath "$dataset_root/data/dvs_processed")"
csv_file="$(realpath "$dataset_root/pathFile.csv")"
for input_path in "$rgb_dir" "$dvs_dir" "$csv_file"; do
  case "$input_path" in
    "$data_root/"*) ;;
    *) echo "Input resolves outside $data_root: $input_path" >&2; exit 1 ;;
  esac
done

raw_args=()
if [[ -d "$dataset_root/data/MAT/img" ]]; then
  raw_dir="$(realpath "$dataset_root/data/MAT/img")"
  case "$raw_dir/" in
    "$data_root/"*) raw_args=(--raw-event-dir "$raw_dir") ;;
    *) echo "Raw events resolve outside $data_root: $raw_dir" >&2; exit 1 ;;
  esac
fi

results_parent="/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs"
resolved_results="$(realpath -m "$results_parent")"
case "$resolved_results/" in
  "$code_root/"*) ;;
  *) echo "Results resolve outside $code_root: $resolved_results" >&2; exit 1 ;;
esac
run_id="${mode}_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$resolved_results/$run_id"
if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 1
fi

echo "Project: $project_root"
echo "RGB: $rgb_dir"
echo "DVS: $dvs_dir"
echo "Raw events: ${raw_dir:-not available}"
echo "Output: $output_dir"
cd "$project_root"
python -m analysis.frequency.stage1_cepdvs \
  --rgb-dir "$rgb_dir" \
  --dvs-dir "$dvs_dir" \
  --csv-dir "$dataset_root" \
  "${raw_args[@]}" \
  --output-dir "$output_dir" \
  --split train --per-class "$per_class" --seed 1000 --size 48 \
  --plot-count "$plot_count"
echo "Summary: $output_dir/summary.md"
