"""Read-only audit of CEP-DVS Edge/DVS training pairs and offline Edge spectra.

Run from the repository root with ``python -m analysis.frequency.audit_cepdvs_transfer``.
The pair order mirrors get_edge2dvs_cepdvs (sorted files, edge-sized split,
prefix ratios, positional/modulo pairing); no model or dataset file is changed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .spectrum import channel_correlation, summarize_frames


def indexed_files(directory: Path, labels: dict[int, int]) -> tuple[list[tuple[int, int, Path]], int]:
    """Replicate the label filter and lexicographic ordering in cepdvs.py."""
    result = []
    skipped = 0
    for name in sorted(name for name in os.listdir(directory) if name.endswith(".pt")):
        sample_id = int(name.split(".")[0])
        if sample_id in labels:
            result.append((sample_id, labels[sample_id], directory / name))
        else:
            skipped += 1
    return result, skipped


def audit_pairs(
    edge: list[tuple[int, int, Path]], dvs: list[tuple[int, int, Path]],
    split_ratio: float, edge_ratio: float, dvs_ratio: float,
) -> tuple[list[dict], dict]:
    """Reproduce training indices, including invalid DVS indices if lists differ."""
    if not 0 < split_ratio < 1 or not 0 < edge_ratio <= 1 or not 0 < dvs_ratio <= 1:
        raise ValueError("Expected 0 < split-ratio < 1 and 0 < source/target ratios <= 1")
    cut = int(len(edge) * split_ratio)
    edge_train = int(cut * edge_ratio)
    dvs_train = int(cut * dvs_ratio)
    if not edge_train or not dvs_train:
        raise ValueError("Training subset is empty")

    records = []
    for index in range(max(edge_train, dvs_train)):
        source = edge[index % edge_train]
        target_index = index % dvs_train
        target = dvs[target_index] if target_index < len(dvs) else None
        records.append({
            "pair_index": index,
            "edge_id": source[0], "edge_label": source[1],
            "dvs_id": target[0] if target else "",
            "dvs_label": target[1] if target else "",
            "id_match": int(source[0] == target[0]) if target else "",
            "label_match": int(source[1] == target[1]) if target else "",
            "valid_target_index": int(target is not None),
        })

    valid = [record for record in records if record["valid_target_index"]]
    edge_train_ids = {item[0] for item in edge[:edge_train]}
    dvs_train_ids = {item[0] for item in dvs[:min(dvs_train, len(dvs))]}
    edge_test_ids = {item[0] for item in edge[cut:]}
    dvs_test = dvs[cut:len(edge)]
    dvs_test_ids = {item[0] for item in dvs_test}
    info = {
        "edge_files": len(edge), "dvs_files": len(dvs),
        "split_index_from_edge_length": cut,
        "edge_train_indices": edge_train, "dvs_train_indices": dvs_train,
        "train_pair_rows": len(records),
        "train_id_matches": sum(row["id_match"] for row in valid),
        "train_label_matches": sum(row["label_match"] for row in valid),
        "train_invalid_target_indices": len(records) - len(valid),
        "dvs_test_indices": len(edge) - cut,
        "dvs_test_invalid_indices": max(0, len(edge) - max(cut, len(dvs))),
        "edge_train_ids_in_dvs_test": len(edge_train_ids & dvs_test_ids),
        "dvs_train_ids_in_edge_test": len(dvs_train_ids & edge_test_ids),
        "edge_ids_without_dvs_file": len({x[0] for x in edge} - {x[0] for x in dvs}),
        "dvs_ids_without_edge_file": len({x[0] for x in dvs} - {x[0] for x in edge}),
    }
    return records, info


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def loader_csv_status(csv_path: Path) -> dict:
    """Check the actual default CSV root without importing the full dataloader."""
    loader_source = Path(__file__).resolve().parents[2] / "dataloader/cepdvs.py"
    root = None
    for node in ast.parse(loader_source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CEPDVS_ROOT" for target in node.targets
        ) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            root = node.value.value
            break
    if root is None:
        return {"loader_csv_path": None, "loader_csv_exists": False,
                "loader_csv_same_sha256": False}
    loader_csv = Path(root) / "pathFile.csv"
    exists = loader_csv.is_file()
    same = (hashlib.sha256(loader_csv.read_bytes()).digest() ==
            hashlib.sha256(csv_path.read_bytes()).digest()) if exists else False
    return {"loader_csv_path": str(loader_csv), "loader_csv_exists": exists,
            "loader_csv_same_sha256": same}


def sample_edge_rows(
    edge: list[tuple[int, int, Path]], sample_ids: list[int],
    prior_metrics: dict[tuple[int, str, int], dict],
) -> tuple[list[dict], list[int]]:
    import torch

    by_id = {item[0]: item for item in edge}
    rows = []
    missing = []
    for sample_id in sample_ids:
        if sample_id not in by_id:
            missing.append(sample_id)
            continue
        _, label, path = by_id[sample_id]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        tensor = saved[0] if isinstance(saved, (tuple, list)) else saved
        stored_label = int(saved[1]) if isinstance(saved, (tuple, list)) and len(saved) > 1 else None
        raw = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if raw.ndim != 3 or raw.shape[0] != 2:
            raise ValueError(f"Expected offline Edge [2,H,W] in {path}, got {raw.shape}")
        if not np.isfinite(raw).all():
            raise ValueError(f"Non-finite offline Edge data in {path}")
        loaded = (raw - 0.5) / 0.5  # EdgeDatasetCEPDVS.__getitem__
        raw_metrics, _, _ = summarize_frames(raw[None])
        loaded_metrics, _, _ = summarize_frames(loaded[None])
        for channel in range(2):
            online = prior_metrics.get((sample_id, "edge_dual_sobel", channel))
            dvs = prior_metrics.get((sample_id, "dvs_processed", channel))
            rows.append({
                "sample_id": sample_id, "label": label,
                "stored_edge_label": stored_label if stored_label is not None else "",
                "stored_label_matches": int(stored_label == label) if stored_label is not None else "",
                "channel": channel,
                "offline_raw_min": float(raw[channel].min()),
                "offline_raw_max": float(raw[channel].max()),
                "offline_loaded_min": float(loaded[channel].min()),
                "offline_loaded_max": float(loaded[channel].max()),
                "offline_raw_activity": raw_metrics[channel]["activity_fraction"],
                "offline_loaded_activity": loaded_metrics[channel]["activity_fraction"],
                "offline_low_fraction": loaded_metrics[channel]["low_fraction"],
                "offline_mid_fraction": loaded_metrics[channel]["mid_fraction"],
                "offline_high_fraction": loaded_metrics[channel]["high_fraction"],
                "online_sobel_high_fraction": float(online["high_fraction"]) if online else "",
                "dvs_high_fraction": float(dvs["high_fraction"]) if dvs else "",
                "offline_channel_correlation": channel_correlation(loaded[None]),
            })
    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Stage-1 smoke run.json; defines fixed sample IDs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside the dataset tree")
    parser.add_argument("--edge-dir", type=Path, help="Default: dataset-root/data/edge")
    parser.add_argument("--dvs-dir", type=Path, help="Default: dataset-root/data/dvs_processed")
    parser.add_argument("--split-ratio", type=float, default=0.9)
    parser.add_argument("--edge-ratio", type=float, default=1.0)
    parser.add_argument("--dvs-ratio", type=float, default=1.0)
    args = parser.parse_args()

    dataset = args.dataset_root.resolve(strict=True)
    edge_dir = (args.edge_dir or dataset / "data/edge").resolve(strict=True)
    dvs_dir = (args.dvs_dir or dataset / "data/dvs_processed").resolve(strict=True)
    reference = args.reference_run.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    for directory in (edge_dir, dvs_dir):
        if not directory.is_dir() or not directory.is_relative_to(dataset):
            raise ValueError(f"Input must resolve to a directory inside dataset-root: {directory}")
    if output.is_relative_to(dataset) or output.exists():
        raise ValueError("Output must be a new directory outside dataset-root")

    from data_process.preprocess_cepdvs_dvs import load_pathfile_csv

    labels = load_pathfile_csv(str(dataset / "pathFile.csv"))
    csv_status = loader_csv_status(dataset / "pathFile.csv")
    prior = json.loads(reference.read_text(encoding="utf-8"))
    if prior.get("dataset") != "CEP-DVS" or not prior.get("manifest"):
        raise ValueError("Reference run.json must contain a CEP-DVS manifest")
    sample_ids = [int(item["sample_id"]) for item in prior["manifest"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Reference manifest has duplicate sample IDs")
    prior_csv = reference.with_name("per_sample_channel.csv")
    with prior_csv.open(newline="", encoding="utf-8") as stream:
        prior_rows = list(csv.DictReader(stream))
    prior_metrics = {(int(row["sample_id"]), row["modality"], int(row["channel"])): row
                     for row in prior_rows}

    edge, edge_skipped = indexed_files(edge_dir, labels)
    dvs, dvs_skipped = indexed_files(dvs_dir, labels)
    pairs, pair_info = audit_pairs(edge, dvs, args.split_ratio, args.edge_ratio, args.dvs_ratio)
    sample_rows, missing = sample_edge_rows(edge, sample_ids, prior_metrics)
    pair_info.update({"edge_files_without_csv_label": edge_skipped,
                      "dvs_files_without_csv_label": dvs_skipped,
                      "reference_samples": len(sample_ids),
                      "offline_edge_samples_found": len(sample_ids) - len(missing),
                      "offline_edge_sample_ids_missing": missing,
                      "stored_edge_label_mismatches": sum(row["stored_label_matches"] == 0
                                                          for row in sample_rows if row["channel"] == 0)})
    pair_info.update(csv_status)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "pair_audit.csv", pairs)
    write_csv(output / "offline_edge_sample_channel.csv", sample_rows)
    metadata = {"dataset_root": str(dataset), "edge_dir": str(edge_dir), "dvs_dir": str(dvs_dir),
                "reference_run": str(reference), "split_ratio": args.split_ratio,
                "edge_ratio": args.edge_ratio, "dvs_ratio": args.dvs_ratio,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pair_audit": pair_info}
    (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    valid = sum(row["valid_target_index"] for row in pairs)
    mismatch = [row for row in pairs if row["valid_target_index"] and not row["id_match"]]
    lines = ["# CEP-DVS transfer input audit", "",
             f"- Edge/DVS labelled files: {len(edge)}/{len(dvs)}",
             f"- Training pairs: {len(pairs)}; valid DVS indices: {valid}",
             f"- Same file ID: {pair_info['train_id_matches']}/{valid}; "
             f"same label: {pair_info['train_label_matches']}/{valid}",
             f"- Invalid DVS test indices: {pair_info['dvs_test_invalid_indices']}",
             f"- Edge samples missing from DVS: {pair_info['edge_ids_without_dvs_file']}; "
             f"DVS samples missing from Edge: {pair_info['dvs_ids_without_edge_file']}",
             f"- Reference samples with offline Edge: {len(sample_ids) - len(missing)}/{len(sample_ids)}",
             f"- Stored offline Edge label mismatches among loaded samples: "
             f"{pair_info['stored_edge_label_mismatches']}",
             f"- Dataloader default pathFile.csv: {csv_status['loader_csv_path']}; "
             f"exists={csv_status['loader_csv_exists']}; "
             f"matches audited CSV={csv_status['loader_csv_same_sha256']}", ""]
    for channel in range(2):
        values = [row for row in sample_rows if row["channel"] == channel]
        if values:
            lines.append(f"- Offline Edge channel {channel}: mean high-band "
                         f"{100 * np.mean([row['offline_high_fraction'] for row in values]):.2f}%; "
                         f"raw active pixels {100 * np.mean([row['offline_raw_activity'] for row in values]):.2f}%")
    if mismatch:
        lines.extend(["", "First ID mismatches (pair index: Edge ID -> DVS ID):"])
        lines.extend(f"- {row['pair_index']}: {row['edge_id']} -> {row['dvs_id']}"
                     for row in mismatch[:10])
    if not csv_status["loader_csv_same_sha256"]:
        lines.extend(["", "WARNING: The default training dataloader CSV is missing or differs. "
                      "Pair results use --dataset-root/pathFile.csv and may not match an unmodified training run."])
    lines.extend(["", "This reproduces the current loader's index rules. It does not train or evaluate a model.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
