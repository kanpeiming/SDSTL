"""Stage-1 spatial spectrum measurements for arrays shaped [T, C, H, W]."""

from __future__ import annotations

import numpy as np


DEFAULT_BANDS = (0.15, 0.35)
RADIAL_BINS = np.linspace(0.0, 1.0 + 1e-12, 33)


def frequency_radius(height: int, width: int) -> np.ndarray:
    """Radial frequency in cycles/pixel, scaled by the diagonal Nyquist radius."""
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    return np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5**2 + 0.5**2)


def spatial_power(frames: np.ndarray, remove_dc: bool = True) -> np.ndarray:
    """Orthonormal 2-D power spectrum of each frame and channel."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 4 or min(frames.shape) == 0:
        raise ValueError("Expected nonempty [T, C, H, W] frames")
    if not np.isfinite(frames).all():
        raise ValueError("Input contains non-finite values")
    if remove_dc:
        frames = frames - frames.mean(axis=(-2, -1), keepdims=True)
    return np.abs(np.fft.fft2(frames, axes=(-2, -1), norm="ortho")) ** 2


def summarize_frames(
    frames: np.ndarray, band_edges: tuple[float, float] = DEFAULT_BANDS
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return per-channel metrics, mean 2-D power, and radial power density.

    Spatial energy is summed per frame, then averaged across frames; zero-power
    frames remain in the denominator rather than being silently dropped.
    """
    low_edge, high_edge = band_edges
    if not 0 < low_edge < high_edge < 1:
        raise ValueError("Band edges must satisfy 0 < low < high < 1")
    frames = np.asarray(frames, dtype=np.float64)
    power = spatial_power(frames, remove_dc=True)
    radius = frequency_radius(frames.shape[-2], frames.shape[-1])
    masks = (
        radius < low_edge,
        (radius >= low_edge) & (radius < high_edge),
        radius >= high_edge,
    )
    channel_power = power.mean(axis=0)
    result = []
    for channel in range(frames.shape[1]):
        band_energy = [float(channel_power[channel, mask].sum()) for mask in masks]
        total = sum(band_energy)
        row = {
            "channel": channel,
            "raw_energy": float(np.square(frames[:, channel]).sum(axis=(-2, -1)).mean()),
            "centered_energy": total,
            "zero_frames": int(np.count_nonzero(np.square(frames[:, channel]).sum(axis=(-2, -1)) == 0)),
            "zero_centered_frames": int(np.count_nonzero(power[:, channel].sum(axis=(-2, -1)) <= 1e-20)),
            "frames": int(frames.shape[0]),
            "activity_fraction": float(np.count_nonzero(np.abs(frames[:, channel]) > 1e-12) / frames[:, channel].size),
            "response_mass": float(frames[:, channel].sum()),
        }
        for name, value in zip(("low", "mid", "high"), band_energy):
            row[f"{name}_energy"] = value
            row[f"{name}_fraction"] = value / total if total > 1e-20 else float("nan")
        result.append(row)
    radial = np.full((frames.shape[1], len(RADIAL_BINS) - 1), np.nan)
    for i in range(len(RADIAL_BINS) - 1):
        mask = (radius >= RADIAL_BINS[i]) & (radius < RADIAL_BINS[i + 1])
        if np.any(mask):
            radial[:, i] = channel_power[:, mask].mean(axis=1)
    return result, channel_power, radial


def channel_correlation(frames: np.ndarray) -> float:
    """Pearson correlation of two channels; undefined for constant channels."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1] != 2:
        raise ValueError("Expected [T, 2, H, W]")
    left, right = frames[:, 0].ravel(), frames[:, 1].ravel()
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
