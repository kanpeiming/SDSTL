"""Read selected metadata from local SDSTL checkpoints without running project code.

Requires PyTorch >= 2.0 for ``weights_only=True``. Prints JSON; never edits inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


FIELDS = (
    "T", "batch_size", "data_set", "dvs_encoding_type", "dvs_sample_ratio",
    "encoder_type", "epoch", "experiment_name", "fine_tuning", "id",
    "img_shape", "lr", "model", "parallel", "pretrained_path", "seed",
    "skip_bottleneck", "skip_classifier", "skip_dvs_input", "skip_features",
    "use_event_attention", "use_eventrpg", "val_strategy", "weight_decay",
    "best_test_acc", "best_train_acc", "best_val_acc",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inspect(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint: {path}")
    states = [(key, checkpoint[key]) for key in ("net", "model_state_dict", "state_dict")
              if isinstance(checkpoint.get(key), dict)]
    if len(states) != 1:
        raise ValueError(f"Expected one model state dictionary in {path}, got {[key for key, _ in states]}")
    state_key, state = states[0]
    tensor_keys = [key for key, value in state.items() if torch.is_tensor(value)]
    top_level = {key: checkpoint[key] for key in FIELDS
                 if key in checkpoint and isinstance(checkpoint[key], (str, int, float, bool, type(None)))}
    sample_shapes = {key: list(state[key].shape) for key in tensor_keys
                     if key.endswith("weight") and
                     any(marker in key for marker in ("dvs_input", "features", "bottleneck", "classifier"))}
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "top_level_keys": list(checkpoint),
        "metadata": top_level,
        "state_key": state_key,
        "tensor_count": len(tensor_keys),
        "tensor_numel": sum(state[key].numel() for key in tensor_keys),
        "first_tensor_keys": tensor_keys[:12],
        "last_tensor_keys": tensor_keys[-12:],
        "weight_shapes": sample_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")
    paths = sorted(directory.glob("*.pth"))
    if not paths:
        parser.error(f"No .pth files in {directory}")
    print(json.dumps([inspect(path) for path in paths], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
