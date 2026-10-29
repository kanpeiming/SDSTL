#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "$script_dir/../.." && pwd -P)"
code_root="$(realpath /home/user/kpm/kpm)"
data_root="$(realpath /home/user/Datasets/kpm-own)"
dataset_root="$(realpath /home/user/Datasets/kpm-own/CEP-DVS)"
results_parent="$(realpath -m /home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs)"

case "$project_root/" in
  "$code_root/"*) ;;
  *) echo "Project resolves outside $code_root: $project_root" >&2; exit 1 ;;
esac
case "$dataset_root/" in
  "$data_root/"*) ;;
  *) echo "Dataset resolves outside $data_root: $dataset_root" >&2; exit 1 ;;
esac
case "$results_parent/" in
  "$code_root/"*) ;;
  *) echo "Results resolve outside $code_root: $results_parent" >&2; exit 1 ;;
esac
for input_path in "$dataset_root/pathFile.csv" "$dataset_root/data/img" \
                  "$dataset_root/data/edge" "$dataset_root/data/dvs_processed"; do
  resolved="$(realpath "$input_path")"
  case "$resolved/" in
    "$data_root/"*) ;;
    *) echo "Input resolves outside $data_root: $resolved" >&2; exit 1 ;;
  esac
done

run_id="loader_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$results_parent/$run_id"
[[ ! -e "$output_dir" ]] || { echo "Output already exists: $output_dir" >&2; exit 1; }
mkdir -p "$output_dir"
cd "$project_root"

python - <<'PY' 2>&1 | tee "$output_dir/check.log"
from pathlib import Path
import torch
from dataloader.cepdvs import (
    CEPDVS_ROOT, CEPDVS_RGB_ROOT, CEPDVS_EDGE_ROOT, CEPDVS_DVS_PROCESSED_ROOT,
    CEPDVSRGBDataset, EdgeDatasetCEPDVS, CEPDVSDVSProcessedDataset,
    get_edge2dvs_cepdvs,
)

expected = Path('/home/user/Datasets/kpm-own/CEP-DVS').resolve()
assert Path(CEPDVS_ROOT).resolve() == expected
for path in (CEPDVS_RGB_ROOT, CEPDVS_EDGE_ROOT, CEPDVS_DVS_PROCESSED_ROOT):
    assert Path(path).resolve().is_relative_to(expected), path
assert (expected / 'pathFile.csv').is_file()

rgb = CEPDVSRGBDataset(CEPDVS_RGB_ROOT)
edge = EdgeDatasetCEPDVS(CEPDVS_EDGE_ROOT)
dvs = CEPDVSDVSProcessedDataset(CEPDVS_DVS_PROCESSED_ROOT)
print('dataset lengths RGB/Edge/DVS:', len(rgb), len(edge), len(dvs))
assert (len(rgb), len(edge), len(dvs)) == (10000, 10000, 10000)
print('one Edge/DVS shape:', tuple(edge[0][0].shape), tuple(dvs[0][0].shape))

train, test = get_edge2dvs_cepdvs(
    batch_size=2, edge_root=CEPDVS_EDGE_ROOT,
    dvs_root=CEPDVS_DVS_PROCESSED_ROOT, num_workers=0,
)
print('split lengths train/test:', len(train.dataset), len(test.dataset))
assert (len(train.dataset), len(test.dataset)) == (9000, 1000)
(source, target), (source_label, target_label) = next(iter(train))
print('batch shapes Edge/DVS:', tuple(source.shape), tuple(target.shape))
print('batch labels Edge/DVS:', source_label.tolist(), target_label.tolist())
assert torch.equal(source_label, target_label)
print('LOADER_SMOKE_OK')
PY

echo "Results: $output_dir"
