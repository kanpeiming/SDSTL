"""Evaluate one unchanged CEP-DVS baseline checkpoint on the current test split.

This is a read-only diagnostic. It uses the same sorted 90/10 split, processed
DVS dataset, VGGSNN class, and time-mean logits as the baseline entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexedTestDataset(Dataset):
    def __init__(self, full_dataset: Dataset, start: int):
        self.full_dataset = full_dataset
        self.start = start

    def __len__(self) -> int:
        return len(self.full_dataset) - self.start

    def __getitem__(self, index: int):
        position = self.start + index
        frames, label = self.full_dataset[position]
        sample_id = int(Path(self.full_dataset.samples[position][0]).stem)
        if tuple(frames.shape) != (10, 2, 48, 48):
            raise ValueError(f"Unexpected DVS shape for ID {sample_id}: {tuple(frames.shape)}")
        return frames, label, sample_id


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    dataset_root = args.dataset_root.resolve(strict=True)
    data_allowed = Path("/home/user/Datasets/kpm-own").resolve(strict=True)
    code_allowed = Path("/home/user/kpm/kpm").resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    if not dataset_root.is_relative_to(data_allowed):
        raise ValueError(f"Dataset outside allowed tree: {dataset_root}")
    if not checkpoint.is_relative_to(code_allowed):
        raise ValueError(f"Checkpoint outside allowed tree: {checkpoint}")
    if not output.is_relative_to(code_allowed):
        raise ValueError(f"Output outside allowed tree: {output}")
    if any(output.iterdir()):
        # The wrapper opens console.log before starting Python; permit only that file.
        if {path.name for path in output.iterdir()} != {"console.log"}:
            raise ValueError(f"Output directory is not fresh: {output}")
    csv_file = (dataset_root / "pathFile.csv").resolve(strict=True)
    dvs_dir = (dataset_root / "data" / "dvs_processed").resolve(strict=True)
    if not csv_file.is_relative_to(data_allowed) or not dvs_dir.is_relative_to(data_allowed):
        raise ValueError("Dataset input resolves outside allowed tree")

    from dataloader.cepdvs import CEPDVSDVSProcessedDataset
    from models.snn_models.VGG import VGGSNN

    full_dataset = CEPDVSDVSProcessedDataset(str(dvs_dir), csv_root=str(dataset_root))
    if len(full_dataset) != 10000:
        raise ValueError(f"Expected 10000 labelled DVS samples, got {len(full_dataset)}")
    test_start = int(len(full_dataset) * 0.9)
    test_dataset = IndexedTestDataset(full_dataset, test_start)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("net"), dict):
        raise ValueError("Expected baseline checkpoint with a 'net' state dict")
    for key, expected in (("data_set", "CEP-DVS"), ("img_shape", 48), ("T", 10),
                          ("pretrained_path", None)):
        if saved.get(key) != expected:
            raise ValueError(f"Unexpected checkpoint {key}: {saved.get(key)!r}")
    if saved.get("use_event_attention", False):
        raise ValueError("This run expects VGGSNN without event attention")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGGSNN(cls_num=20, img_shape=48, use_event_attention=False)
    model.load_state_dict(saved["net"], strict=True)
    model = model.to(device).eval()
    predictions = []
    class_total = [0] * 20
    class_correct = [0] * 20

    with torch.inference_mode():
        for frames, labels, sample_ids in loader:
            frames = frames.to(device, dtype=torch.float32)
            if not torch.isfinite(frames).all():
                raise ValueError("Non-finite DVS input")
            logits = model(frames)
            expected_shape = (frames.shape[0], 10, 20)
            if tuple(logits.shape) != expected_shape or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid logits shape/values: {tuple(logits.shape)}")
            probabilities = logits.mean(dim=1).softmax(dim=1).cpu()
            values, predicted = probabilities.max(dim=1)
            for sample_id, label, pred, confidence in zip(
                    sample_ids.tolist(), labels.tolist(), predicted.tolist(), values.tolist()):
                if not 0 <= label < 20:
                    raise ValueError(f"Invalid label for ID {sample_id}: {label}")
                correct = int(label == pred)
                class_total[label] += 1
                class_correct[label] += correct
                predictions.append({"sample_id": sample_id, "label": label,
                                    "prediction": pred, "correct": correct,
                                    "confidence": confidence})

    if len(predictions) != 1000 or len({row["sample_id"] for row in predictions}) != 1000:
        raise ValueError("Test set is not 1000 unique samples")
    write_csv(output / "predictions.csv", predictions)
    per_class = [{"label": label, "count": class_total[label],
                  "correct": class_correct[label],
                  "accuracy": class_correct[label] / class_total[label] if class_total[label] else None}
                 for label in range(20)]
    write_csv(output / "per_class.csv", per_class)
    correct = sum(row["correct"] for row in predictions)
    import dataloader.cepdvs as loader_source

    metadata = {
        "dataset_root": str(dataset_root), "dvs_dir": str(dvs_dir),
        "csv_sha256": sha256(csv_file), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": saved.get("epoch"),
        "checkpoint_pretrained_path": saved.get("pretrained_path"),
        "split": "last 10 percent of sorted processed DVS files",
        "test_start": test_start, "test_samples": len(predictions),
        "batch_size": args.batch_size, "device": str(device),
        "torch": torch.__version__,
        "source_sha256": {
            "eval_cepdvs_baseline.py": sha256(Path(__file__)),
            "dataloader/cepdvs.py": sha256(Path(loader_source.__file__)),
            "models/snn_models/VGG.py": sha256(Path(VGGSNN.__module__.replace(".", "/") + ".py")),
        },
        "correct": correct, "accuracy": correct / len(predictions),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = ("# CEP-DVS current-test baseline evaluation\n\n"
               f"- Checkpoint: {checkpoint.name}\n"
               f"- Current dataset: {dataset_root}\n"
               f"- Test samples: {len(predictions)} (sorted indices {test_start}..{len(full_dataset)-1})\n"
               f"- Accuracy: {correct}/{len(predictions)} = {correct / len(predictions):.4f}\n"
               f"- Device: {device}\n"
               "- No training or parameter updates.\n")
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
