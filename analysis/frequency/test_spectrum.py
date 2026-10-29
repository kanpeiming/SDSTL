"""Checks for the measurements that determine stage-1 conclusions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .spectrum import channel_correlation, spatial_power, summarize_frames
from .stage1_cepdvs import choose_samples, raw_event_frames


class SpectrumTests(unittest.TestCase):
    def test_parseval_and_dc_removal(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(3, 2, 16, 24)) + 3
        centered = data - data.mean(axis=(-2, -1), keepdims=True)
        power = spatial_power(data)
        np.testing.assert_allclose(power.sum(axis=(-2, -1)),
                                   np.square(centered).sum(axis=(-2, -1)), rtol=1e-12)

    def test_zero_and_constant_frames_are_reported(self):
        data = np.stack((np.zeros((1, 16, 16)), np.ones((1, 16, 16))))
        rows, _, _ = summarize_frames(data)
        self.assertEqual(rows[0]["zero_frames"], 1)
        self.assertEqual(rows[0]["zero_centered_frames"], 2)
        self.assertTrue(np.isnan(rows[0]["high_fraction"]))

    def test_known_low_and_high_spatial_frequencies(self):
        x = np.arange(32)
        low = np.broadcast_to(np.cos(2 * np.pi * x / 32), (32, 32))
        high = np.broadcast_to(np.cos(2 * np.pi * 10 * x / 32), (32, 32))
        low_row = summarize_frames(low[None, None])[0][0]
        high_row = summarize_frames(high[None, None])[0][0]
        self.assertGreater(low_row["low_fraction"], 0.999)
        self.assertGreater(high_row["high_fraction"], 0.999)

    def test_channel_correlation_handles_identical_and_constant(self):
        rng = np.random.default_rng(9)
        signal = rng.random((2, 8, 8))
        duplicate = np.stack((signal, signal), axis=1)
        self.assertAlmostEqual(channel_correlation(duplicate), 1.0)
        self.assertTrue(np.isnan(channel_correlation(np.zeros_like(duplicate))))

    def test_selection_uses_sorted_dvs_train_split_and_existing_rgb(self):
        with TemporaryDirectory() as temp:
            rgb, dvs = Path(temp) / "rgb", Path(temp) / "dvs"
            rgb.mkdir()
            dvs.mkdir()
            for sample_id in range(10):
                (dvs / f"{sample_id:05d}.pt").touch()
                (rgb / f"{sample_id:05d}.jpg").touch()
            labels = {sample_id: sample_id % 2 for sample_id in range(10)}
            first = choose_samples(rgb, dvs, labels, "train", 3, 1000)
            second = choose_samples(rgb, dvs, labels, "train", 3, 1000)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 6)
            self.assertTrue(all(sample_id < 9 for sample_id, _ in first))
            self.assertEqual(choose_samples(rgb, dvs, labels, "test", 3, 1000), [(9, 1)])

    def test_raw_event_binning_tracks_polarity_and_last_bin(self):
        events = {"x": [2, 2, 3, -1], "y": [4, 4, 5, 0],
                  "p": [0, 1, 1, 0], "ts": [10, 11, 20, 20]}
        counts, by_bin, invalid = raw_event_frames(events, 4)
        self.assertEqual(invalid, 1)
        self.assertEqual(counts[0, 0, 4, 2], 1)
        self.assertEqual(counts[0, 1, 4, 2], 1)
        self.assertEqual(counts[3, 1, 5, 3], 1)
        self.assertEqual(by_bin.sum(), 3)


if __name__ == "__main__":
    unittest.main()
