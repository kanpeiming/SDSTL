"""Inspect CEP-DVS RGB, actual dual-Sobel edge, and event frames by file ID.

Run from the SDSTL repository root with ``python -m analysis.frequency.stage1_cepdvs``.
This script reads datasets, writes only to --output-dir, and never trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .spectrum import DEFAULT_BANDS, RADIAL_BINS, channel_correlation, summarize_frames


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True, help="CEP-DVS data/img")
    parser.add_argument("--dvs-dir", type=Path, required=True, help="CEP-DVS data/dvs_processed")
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing pathFile.csv")
    parser.add_argument("--raw-event-dir", type=Path, help="Optional CEP-DVS data/MAT/img")
    parser.add_argument("--output-dir", type=Path, required=True, help="New results directory")
    parser.add_argument("--split", choices=("train", "test", "all"), default="train",
                        help="DVS baseline split: first 90 percent of sorted .pt files")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--plot-count", type=int, default=3)
    parser.add_argument("--low-edge", type=float, default=DEFAULT_BANDS[0])
    parser.add_argument("--high-edge", type=float, default=DEFAULT_BANDS[1])
    return parser.parse_args()


def choose_samples(
    rgb_dir: Path, dvs_dir: Path, labels: dict[int, int], split: str,
    per_class: int, seed: int
) -> list[tuple[int, int]]:
    if per_class < 1:
        raise ValueError("--per-class must be at least 1")
    paths = sorted(dvs_dir.glob("*.pt"))
    cut = int(len(paths) * 0.9)
    if split == "train":
        paths = paths[:cut]
    elif split == "test":
        paths = paths[cut:]
    by_class: dict[int, list[int]] = defaultdict(list)
    for path in paths:
        if not path.stem.isdecimal():
            continue
        sample_id = int(path.stem)
        if sample_id in labels and (rgb_dir / f"{sample_id:05d}.jpg").is_file():
            by_class[labels[sample_id]].append(sample_id)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(by_class):
        ids = np.asarray(sorted(set(by_class[label])), dtype=int)
        picks = rng.choice(ids, size=min(per_class, len(ids)), replace=False)
        selected.extend((int(sample_id), int(label)) for sample_id in sorted(picks))
    return selected


def raw_event_frames(events: dict, time_bins: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Count valid raw events in the two polarity channels and time bins."""
    count = np.zeros((time_bins, 2, 180, 240), dtype=np.float32)
    x = np.asarray(events["x"]).ravel().astype(np.int64)
    y = np.asarray(events["y"]).ravel().astype(np.int64)
    p = np.asarray(events["p"]).ravel().astype(np.int64)
    ts = np.asarray(events["ts"]).ravel().astype(np.float64)
    if not (len(x) == len(y) == len(p) == len(ts)):
        raise ValueError("Raw event coordinates, polarity, and timestamps have different lengths")
    if len(ts) == 0:
        return count, np.zeros((time_bins, 2), dtype=np.int64), 0
    span = float(ts.max() - ts.min())
    bins = np.floor((ts - ts.min()) / span * (time_bins - 1)).astype(np.int64) if span > 0 else np.zeros(len(ts), dtype=np.int64)
    valid = ((x >= 0) & (x < 240) & (y >= 0) & (y < 180) &
             (p >= 0) & (p < 2) & (bins >= 0) & (bins < time_bins))
    np.add.at(count, (bins[valid], p[valid], y[valid], x[valid]), 1)
    per_bin_polarity = count.sum(axis=(2, 3)).astype(np.int64)
    return count, per_bin_polarity, int(np.count_nonzero(~valid))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_examples(
    output_dir: Path, sample_id: int, rgb: np.ndarray, edge: np.ndarray,
    dvs: np.ndarray, edge_power: np.ndarray, dvs_power: np.ndarray
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = dvs.shape[0] // 2
    images = (
        rgb.transpose(1, 2, 0), edge[0, 0],
        dvs[center, 0], dvs[center, 1],
        np.log1p(np.fft.fftshift(edge_power[0])),
        np.log1p(np.fft.fftshift(dvs_power[0])),
    )
    titles = ("RGB [0,1]", "Sobel (one of two identical channels)",
              "DVS channel 0, middle bin", "DVS channel 1, middle bin",
              "Sobel log1p centered power", "DVS ch0 log1p centered power")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, picture, title in zip(axes.flat, images, titles):
        axis.imshow(np.clip(picture, 0, 1) if title.startswith("RGB") else picture,
                    cmap=None if title.startswith("RGB") else "gray")
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"example_{sample_id:05d}.png", dpi=135)
    plt.close(fig)


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["modality"], row["label"], row["channel"])].append(row)
    output = []
    numeric = ("raw_energy", "centered_energy", "activity_fraction", "response_mass",
               "zero_frames", "zero_centered_frames", "low_fraction", "mid_fraction", "high_fraction")
    for (modality, label, channel), group in sorted(groups.items()):
        result = {"modality": modality, "label": label, "channel": channel, "n_samples": len(group)}
        for name in numeric:
            values = np.asarray([row[name] for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            result[name + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            result[name + "_std"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            result[name + "_n_valid"] = int(finite.size)
        output.append(result)
    return output


def source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    result = {}
    for name in ("stage1_cepdvs.py", "spectrum.py"):
        path = Path(__file__).with_name(name)
        result[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, command in (("git_commit", ("git", "rev-parse", "HEAD")),
                          ("git_status", ("git", "status", "--short"))):
        try:
            process = subprocess.run(command, cwd=root, check=False, capture_output=True,
                                     text=True, timeout=5)
            result[name] = process.stdout.strip() if process.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[name] = None
    return result


def summary_markdown(metric_rows: list[dict], auxiliary_rows: list[dict]) -> str:
    """A compact numeric digest for inspecting a remote run; no causal claims."""
    selected = (
        ("rgb_gray_pixel", 0),
        ("edge_dual_sobel", 0),
        ("dvs_processed", 0),
        ("dvs_processed", 1),
        ("dvs_native_count", 0),
        ("dvs_native_count", 1),
    )
    lines = ["# Stage 1 numeric digest", "",
             "All rows are sample averages; band fractions omit zero-centered-energy samples.",
             "Original RGB/Edge and DVS amplitudes have different meanings. This table is descriptive.", "",
             "| Modality | Channel | Samples | Valid spectra | Low % | Mid % | High % | Activity % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for modality, channel in selected:
        group = [row for row in metric_rows if row["modality"] == modality and row["channel"] == channel]
        if not group:
            continue
        valid = [row for row in group if np.isfinite(row["low_fraction"])]
        def mean_percent(key: str, records: list[dict]) -> str:
            return f"{100 * np.mean([row[key] for row in records]):.2f}" if records else "NA"
        lines.append(f"| {modality} | {channel} | {len(group)} | {len(valid)} | "
                     f"{mean_percent('low_fraction', valid)} | {mean_percent('mid_fraction', valid)} | "
                     f"{mean_percent('high_fraction', valid)} | {mean_percent('activity_fraction', group)} |")
    lines.extend(["", "## Input checks", ""])
    edge_corr = [row["edge_channel_correlation"] for row in auxiliary_rows
                 if np.isfinite(row["edge_channel_correlation"])]
    dvs_corr = [row["dvs_channel_correlation"] for row in auxiliary_rows
                if np.isfinite(row["dvs_channel_correlation"])]
    raw_rows = [row for row in auxiliary_rows if row["raw_event_count"] is not None]
    lines.append(f"- Edge channel correlation (valid samples): {float(np.mean(edge_corr)):.6f}" if edge_corr
                 else "- Edge channel correlation: undefined for all sampled images")
    lines.append(f"- DVS channel correlation (valid samples): {float(np.mean(dvs_corr)):.6f}" if dvs_corr
                 else "- DVS channel correlation: undefined for all sampled images")
    lines.append(f"- Samples with raw event files: {len(raw_rows)}/{len(auxiliary_rows)}")
    if raw_rows:
        errors = [row["rebuilt_max_abs_error"] for row in raw_rows]
        lines.append(f"- Rebuilt vs saved DVS maximum absolute error: {max(errors):.6g}")
        last_shares = []
        for row in raw_rows:
            counts = np.asarray(json.loads(row["raw_event_count_per_bin_polarity"]), dtype=float)
            if counts.sum() > 0:
                last_shares.append(float(counts[-1].sum() / counts.sum()))
        lines.append(f"- Raw events in last time bin, mean sample share: {100 * np.mean(last_shares):.3f}%"
                     if last_shares else "- Raw events in last time bin: no valid events")
    lines.extend(["", "These measurements alone do not establish whether a frequency band helps classification.", ""])
    return "\n".join(lines)


def main() -> None:
    args = arguments()
    if args.size < 8 or args.plot_count < 0:
        raise ValueError("--size must be >= 8 and --plot-count must be >= 0")
    if not 0 < args.low_edge < args.high_edge < 1:
        raise ValueError("Expected 0 < low-edge < high-edge < 1")
    for path in (args.rgb_dir, args.dvs_dir, args.csv_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.raw_event_dir is not None and not args.raw_event_dir.is_dir():
        raise FileNotFoundError(args.raw_event_dir)
    from data_process.preprocess_cepdvs_dvs import events_to_frames, load_mat_file, load_pathfile_csv
    from pretrain.Edge import SobelEdgeExtractionModule
    import torch

    labels = load_pathfile_csv(str(args.csv_dir / "pathFile.csv"))
    selected = choose_samples(args.rgb_dir, args.dvs_dir, labels, args.split,
                              args.per_class, args.seed)
    if not selected:
        raise RuntimeError("No paired RGB/DVS samples with labels found")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sobel = SobelEdgeExtractionModule(device="cpu", in_channels=3).eval()
    torch_mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    torch_std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    metric_rows: list[dict] = []
    auxiliary_rows: list[dict] = []
    radial_sum: dict[tuple[str, int], np.ndarray] = {}
    radial_count: dict[tuple[str, int], np.ndarray] = {}
    manifest: list[dict] = []
    for ordinal, (sample_id, label) in enumerate(selected):
        rgb_path = args.rgb_dir / f"{sample_id:05d}.jpg"
        dvs_path = args.dvs_dir / f"{sample_id:05d}.pt"
        with Image.open(rgb_path) as image:
            bilinear = getattr(Image, "Resampling", Image).BILINEAR
            resized = image.convert("RGB").resize((args.size, args.size), bilinear)
            rgb = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        rgb_tensor = torch.from_numpy(rgb.copy())
        normalized = (rgb_tensor - torch_mean) / torch_std
        with torch.no_grad():
            edge_single = sobel(normalized.unsqueeze(0))[0]
            edge = torch.cat((edge_single, edge_single), dim=0).numpy()[None]
        dvs_record = torch.load(dvs_path, map_location="cpu", weights_only=True)
        dvs_tensor = dvs_record[0] if isinstance(dvs_record, (tuple, list)) else dvs_record
        if isinstance(dvs_record, (tuple, list)) and len(dvs_record) > 1:
            saved_label = int(dvs_record[1])
            if saved_label != label:
                raise ValueError(f"DVS label differs from pathFile.csv for {sample_id:05d}: {saved_label} != {label}")
        dvs = dvs_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if dvs.ndim != 4 or dvs.shape[1] != 2:
            raise ValueError(f"Expected [T, 2, H, W] in {dvs_path}: {dvs.shape}")
        if dvs.shape[-2:] != (args.size, args.size):
            raise ValueError(f"Processed DVS size {dvs.shape[-2:]} does not match --size {args.size}")
        modalities = {
            "rgb_pixel": rgb[None],
            "rgb_model_normalized": normalized.numpy()[None],
            "rgb_gray_pixel": (rgb * np.asarray([0.299, 0.587, 0.114])[:, None, None]).sum(axis=0)[None, None],
            "edge_dual_sobel": edge,
            "dvs_processed": dvs,
            "dvs_sum": dvs.sum(axis=1, keepdims=True),
            "dvs_difference": (dvs[:, 1] - dvs[:, 0])[:, None],
        }
        raw_path = args.raw_event_dir / f"{sample_id:05d}.mat" if args.raw_event_dir else None
        raw_event_count = None
        invalid_event_count = None
        per_bin_polarity = None
        if raw_path is not None and raw_path.is_file():
            events = load_mat_file(str(raw_path))
            raw_event_count = int(np.size(events["ts"]))
            raw_count, per_bin_polarity, invalid_event_count = raw_event_frames(events, dvs.shape[0])
            modalities["dvs_native_count"] = raw_count
            native = events_to_frames(events, height=180, width=240, time_bins=dvs.shape[0])
            modalities["dvs_native_normalized"] = native.numpy()
            resized_native = torch.nn.functional.interpolate(
                native.float(), size=(args.size, args.size), mode="bilinear", align_corners=False
            ).numpy()
            modalities["dvs_rebuilt_processed"] = resized_native
            rebuild_error = float(np.max(np.abs(resized_native - dvs)))
        else:
            rebuild_error = None
        powers = {}
        for name, frames in modalities.items():
            rows, mean_power, radial = summarize_frames(frames, (args.low_edge, args.high_edge))
            powers[name] = mean_power
            for row in rows:
                metric_rows.append({"sample_id": sample_id, "label": label, "modality": name,
                                    "height": frames.shape[-2], "width": frames.shape[-1], **row})
                key = (name, row["channel"])
                valid = np.isfinite(radial[row["channel"]])
                if key not in radial_sum:
                    radial_sum[key] = np.zeros(radial.shape[1], dtype=float)
                    radial_count[key] = np.zeros(radial.shape[1], dtype=int)
                radial_sum[key][valid] += radial[row["channel"], valid]
                radial_count[key][valid] += 1
        auxiliary_rows.append({"sample_id": sample_id, "label": label,
                               "edge_channel_correlation": channel_correlation(edge),
                               "dvs_channel_correlation": channel_correlation(dvs),
                               "dvs_bin_mass": json.dumps(dvs.sum(axis=(1, 2, 3)).tolist()),
                               "dvs_channel_bin_mass": json.dumps(dvs.sum(axis=(2, 3)).tolist()),
                               "raw_event_count": raw_event_count,
                               "raw_invalid_event_count": invalid_event_count,
                               "raw_event_count_per_bin_polarity": json.dumps(per_bin_polarity.tolist()) if per_bin_polarity is not None else "",
                               "rebuilt_max_abs_error": rebuild_error})
        manifest.append({"sample_id": sample_id, "label": label, "rgb": str(rgb_path),
                         "dvs": str(dvs_path), "raw_events": str(raw_path) if raw_path and raw_path.is_file() else None})
        if ordinal < args.plot_count:
            plot_examples(args.output_dir, sample_id, rgb, edge, dvs,
                          powers["edge_dual_sobel"], powers["dvs_processed"])
        print(f"[{ordinal + 1}/{len(selected)}] sample {sample_id:05d}, class {label}", flush=True)
    save_csv(args.output_dir / "per_sample_channel.csv", metric_rows)
    save_csv(args.output_dir / "per_class_channel.csv", aggregate_metrics(metric_rows))
    save_csv(args.output_dir / "sample_auxiliary.csv", auxiliary_rows)
    radial_rows = []
    for (modality, channel), values in sorted(radial_sum.items()):
        count = radial_count[(modality, channel)]
        for index in range(len(count)):
            radial_rows.append({"modality": modality, "channel": channel,
                                "radius_start": RADIAL_BINS[index], "radius_end": RADIAL_BINS[index + 1],
                                "power_density_mean": values[index] / count[index] if count[index] else float("nan"),
                                "n_samples": count[index]})
    save_csv(args.output_dir / "radial_power_density.csv", radial_rows)
    (args.output_dir / "summary.md").write_text(summary_markdown(metric_rows, auxiliary_rows), encoding="utf-8")
    metadata = {"dataset": "CEP-DVS", "split": args.split, "seed": args.seed,
                "per_class": args.per_class, "n_samples": len(selected),
                "size": args.size, "band_edges_normalized_radius": [args.low_edge, args.high_edge],
                "fft_normalization": "ortho", "dc_removed_for_band_energy": True,
                "radial_units": "cycles/pixel divided by diagonal Nyquist radius",
                "rgb_normalize_mean": MEAN, "rgb_normalize_std": STD,
                "edge": "same Sobel gradient magnitude repeated in two channels",
                "dvs_channel_order": "channel 0 negative, channel 1 positive (verify upstream dataset)",
                "raw_event_path_supplied": args.raw_event_dir is not None,
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "source": source_provenance(), "manifest": manifest}
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(selected)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
