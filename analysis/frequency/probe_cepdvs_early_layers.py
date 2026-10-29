"""Probe one CEP-DVS baseline before and after its first two shared pool calls.

Hooks observe Conv, BN, LIF output, and pool output without changing forward.
Pre-spike membrane traces are reconstructed from BN input and observed spikes.
Only per-sample summary statistics are saved; feature tensors are not persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .spectrum import DEFAULT_BANDS, frequency_radius, spatial_power


BASELINE_NAME = "SDSTL_CEP-DVS_TET_img48_T10_seed1000_dvs1_enc-TET_lr0.001_VGGSNN_namebaseline.pth"
STAGES = ("input", "dvs_conv", "dvs_bn", "dvs_mem", "dvs_spike",
          "feature0_conv", "feature0_bn", "feature0_mem", "feature0_spike", "pool1",
          "feature2_conv", "feature2_bn", "feature2_mem", "feature2_spike",
          "feature3_conv", "feature3_bn", "feature3_mem", "feature3_spike", "pool2")
SPIKE_STAGES = {"dvs_spike", "feature0_spike", "feature2_spike", "feature3_spike"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def add_layer_hooks(model):
    """Capture one-sample outputs; the shared pool is tracked by call order."""
    captured = {}
    pool_calls = []
    handles = []

    def hook(name: str, flattened: bool = False):
        def observe(_module, _inputs, output):
            tensor = output.detach()
            if flattened:
                tensor = tensor.reshape(1, 10, *tensor.shape[1:])
            captured[name] = tensor[0].cpu().clone()
        return observe

    for name, layer in (("dvs", model.dvs_input),
                        ("feature0", model.features[0]),
                        ("feature2", model.features[2]),
                        ("feature3", model.features[3])):
        handles.append(layer.fwd.module[0].register_forward_hook(hook(f"{name}_conv", True)))
        handles.append(layer.fwd.module[1].register_forward_hook(hook(f"{name}_bn", True)))
        handles.append(layer.act.register_forward_hook(hook(f"{name}_spike")))

    pool = model.features[1]
    if not (pool is model.features[4] is model.features[7] is model.features[10]):
        raise ValueError("Expected four calls to the same pool module")

    def pool_hook(_module, _inputs, output):
        pool_calls.append(output.detach()[0].cpu().clone())

    handles.append(pool.register_forward_hook(pool_hook))
    return captured, pool_calls, handles


def membrane_trace(bn: torch.Tensor, spikes: torch.Tensor, layer) -> tuple[torch.Tensor, int]:
    """Replay the local LIF recurrence without altering the model."""
    membrane = torch.zeros_like(bn[0])
    traces = []
    mismatch = 0
    for t in range(bn.shape[0]):
        pre_spike = membrane * layer.act.tau + bn[t]
        expected = (pre_spike > layer.act.thresh).to(spikes.dtype)
        mismatch += int(torch.count_nonzero(expected != spikes[t]).item())
        traces.append(pre_spike)
        membrane = (1 - spikes[t]) * pre_spike
    return torch.stack(traces), mismatch


def capture_one(model, frames: torch.Tensor) -> tuple[torch.Tensor, dict, int]:
    if tuple(frames.shape[:3]) != (1, 10, 2):
        raise ValueError(f"Expected one sample [1,10,2,H,W], got {tuple(frames.shape)}")
    captured, pool_calls, handles = add_layer_hooks(model)
    try:
        output = model(frames)
    finally:
        for handle in handles:
            handle.remove()
    if len(pool_calls) != 4:
        raise ValueError(f"Expected 4 shared-pool calls, saw {len(pool_calls)}")
    captured["input"] = frames.detach()[0].cpu().clone()
    captured["pool1"], captured["pool2"] = pool_calls[:2]
    mismatch = 0
    for name, layer in (("dvs", model.dvs_input),
                        ("feature0", model.features[0]),
                        ("feature2", model.features[2]),
                        ("feature3", model.features[3])):
        trace, count = membrane_trace(captured[f"{name}_bn"],
                                      captured[f"{name}_spike"], layer)
        captured[f"{name}_mem"] = trace
        mismatch += count
    if set(captured) != set(STAGES):
        raise ValueError(f"Missing or extra stages: {set(STAGES) ^ set(captured)}")
    return output, captured, mismatch


def describe(frames: torch.Tensor, stage: str) -> dict:
    array = frames.numpy().astype(np.float64, copy=False)
    if array.ndim != 4 or not np.isfinite(array).all():
        raise ValueError(f"Invalid stage tensor: {stage}")
    time_bins, channels, height, width = array.shape
    centered = array - array.mean(axis=(-2, -1), keepdims=True)
    power = spatial_power(array)
    radius = frequency_radius(height, width)
    low_edge, high_edge = DEFAULT_BANDS
    band_energies = [float(power[..., mask].sum()) for mask in
                     (radius < low_edge, (radius >= low_edge) & (radius < high_edge),
                      radius >= high_edge)]
    total = sum(band_energies)
    if total <= 1e-20:
        fractions = [float("nan")] * 3
    else:
        fractions = [value / total for value in band_energies]
    spike_rate = float(array.mean()) if stage in SPIKE_STAGES else ""
    return {"stage": stage, "time_bins": time_bins, "channels": channels,
            "height": height, "width": width,
            "rms": float(np.sqrt(np.mean(array * array))),
            "centered_rms": float(np.sqrt(np.mean(centered * centered))),
            "activity_fraction": float(np.mean(np.abs(array) > 1e-12)),
            "zero_centered_frame_channel_fraction": float(np.mean(power.sum(axis=(-2, -1)) <= 1e-20)),
            "low_fraction": fractions[0], "mid_fraction": fractions[1],
            "high_fraction": fractions[2], "spike_rate": spike_rate}


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data_allowed = Path("/home/user/Datasets/kpm-own").resolve(strict=True)
    code_allowed = Path("/home/user/kpm/kpm").resolve(strict=True)
    dataset = args.dataset_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    reference = args.reference_run.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    if not dataset.is_relative_to(data_allowed):
        raise ValueError("Dataset outside approved data root")
    if not all(path.is_relative_to(code_allowed) for path in (checkpoint, reference, output)):
        raise ValueError("Checkpoint/reference/output outside approved code root")
    if checkpoint.name != BASELINE_NAME:
        raise ValueError(f"Unexpected checkpoint: {checkpoint.name}")
    for path in (dataset / "pathFile.csv", dataset / "data/dvs_processed"):
        if not path.resolve(strict=True).is_relative_to(data_allowed):
            raise ValueError(f"Input outside approved data root: {path}")
    if {path.name for path in output.iterdir()} not in (set(), {"console.log"}):
        raise ValueError("Output directory must be new and contain at most console.log")

    reference_info = json.loads(reference.read_text(encoding="utf-8"))
    if reference_info.get("dataset") != "CEP-DVS" or reference_info.get("split") != "train":
        raise ValueError("Expected stage-1 CEP-DVS training manifest")
    chosen = {}
    for item in reference_info["manifest"]:
        label = int(item["label"])
        chosen.setdefault(label, int(item["sample_id"]))
    if len(chosen) != 20:
        raise ValueError(f"Expected 20 classes, got {len(chosen)}")

    from data_process.preprocess_cepdvs_dvs import load_pathfile_csv
    from models.snn_models.VGG import VGGSNN

    labels = load_pathfile_csv(str(dataset / "pathFile.csv"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("net"), dict):
        raise ValueError("Expected baseline checkpoint with net state")
    for key, expected in (("data_set", "CEP-DVS"), ("img_shape", 48),
                          ("T", 10), ("pretrained_path", None)):
        if state.get(key) != expected:
            raise ValueError(f"Unexpected checkpoint {key}: {state.get(key)!r}")
    if state.get("use_event_attention", False):
        raise ValueError("Expected baseline without event attention")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGGSNN(cls_num=20, img_shape=48, use_event_attention=False).to(device)
    model.load_state_dict(state["net"], strict=True)
    model.eval()

    rows = []
    predictions = []
    max_logit_difference = 0.0
    membrane_mismatches = 0
    with torch.inference_mode():
        for label, sample_id in sorted(chosen.items()):
            if labels.get(sample_id) != label:
                raise ValueError(f"CSV label mismatch for ID {sample_id}")
            file = (dataset / "data/dvs_processed" / f"{sample_id:05d}.pt").resolve(strict=True)
            if not file.is_relative_to(data_allowed):
                raise ValueError(f"DVS file outside approved root: {file}")
            saved = torch.load(file, map_location="cpu", weights_only=True)
            if isinstance(saved, (tuple, list)) and len(saved) > 1 and int(saved[1]) != label:
                raise ValueError(f"Stored DVS label mismatch for ID {sample_id}")
            frames = saved[0] if isinstance(saved, (tuple, list)) else saved
            if tuple(frames.shape) != (10, 2, 48, 48):
                raise ValueError(f"Unexpected sample shape for ID {sample_id}: {tuple(frames.shape)}")
            if not torch.isfinite(frames).all():
                raise ValueError(f"Non-finite DVS input for ID {sample_id}")
            batch = frames[None].to(device, dtype=torch.float32)
            reference_logits = model(batch)
            hooked_logits, captured, mismatch = capture_one(model, batch)
            difference = float((reference_logits - hooked_logits).abs().max().item())
            if difference > 1e-6 or not torch.isfinite(hooked_logits).all():
                raise ValueError(f"Hook changed logits for ID {sample_id}: {difference}")
            max_logit_difference = max(max_logit_difference, difference)
            membrane_mismatches += mismatch
            predicted = int(hooked_logits.mean(dim=1).argmax(dim=1).item())
            predictions.append({"sample_id": sample_id, "label": label,
                                "prediction": predicted, "correct": int(label == predicted),
                                "hooked_logit_max_abs_difference": difference,
                                "membrane_spike_mismatch": mismatch})
            for stage in STAGES:
                rows.append({"sample_id": sample_id, "label": label,
                             **describe(captured[stage], stage)})
            print(f"[{len(predictions)}/20] ID {sample_id:05d} class {label}", flush=True)

    save_csv(output / "per_sample_stage.csv", rows)
    save_csv(output / "predictions.csv", predictions)
    digest = {"dataset_root": str(dataset), "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "reference_run": str(reference),
              "reference_run_sha256": sha256(reference), "source_sha256": sha256(Path(__file__)),
              "selected_ids": [row["sample_id"] for row in predictions],
              "model": "VGGSNN", "T": 10, "device": str(device),
              "torch": torch.__version__, "band_edges_normalized_radius": list(DEFAULT_BANDS),
              "frequency_measure": "per-sample pooled channel/time centered FFT energy",
              "max_hook_logit_difference": max_logit_difference,
              "membrane_spike_mismatches": membrane_mismatches}
    (output / "run.json").write_text(json.dumps(digest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# CEP-DVS baseline early-layer probe", "",
             f"- Samples: {len(predictions)} (one fixed training ID per class)",
             f"- Correct predictions: {sum(row['correct'] for row in predictions)}/{len(predictions)}",
             f"- Maximum hooked/unhooked logit difference: {max_logit_difference:g}",
             f"- Reconstructed membrane/spike mismatches: {membrane_mismatches}", "",
             "| Stage | Shape T x C x H x W | RMS | Centered RMS | High % | Active % | Spike rate % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for stage in STAGES:
        group = [row for row in rows if row["stage"] == stage]
        first = group[0]
        shape = " x ".join(str(first[key]) for key in ("time_bins", "channels", "height", "width"))
        def average(key):
            values = [row[key] for row in group if isinstance(row[key], (int, float)) and math.isfinite(row[key])]
            return sum(values) / len(values) if values else float("nan")
        rate = f"{100 * average('spike_rate'):.2f}" if stage in SPIKE_STAGES else "-"
        lines.append(f"| {stage} | {shape} | {average('rms'):.4g} | "
                     f"{average('centered_rms'):.4g} | {100 * average('high_fraction'):.2f} | "
                     f"{100 * average('activity_fraction'):.2f} | {rate} |")
    lines.extend(["", "Different spatial resolutions use the same normalized-radius convention,",
                  "so changes in fractions are descriptive and are not strict transfer gains.",
                  "Only retained-spatial tensors are analyzed; no FFT is applied after flattening.", ""])
    summary = "\n".join(lines)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
