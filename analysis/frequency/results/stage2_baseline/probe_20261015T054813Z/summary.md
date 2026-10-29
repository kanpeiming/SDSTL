# CEP-DVS baseline early-layer probe

- Samples: 20 (one fixed training ID per class)
- Correct predictions: 20/20
- Maximum hooked/unhooked logit difference: 0
- Reconstructed membrane/spike mismatches: 0

| Stage | Shape T x C x H x W | RMS | Centered RMS | High % | Active % | Spike rate % |
|---|---|---:|---:|---:|---:|---:|
| input | 10 x 2 x 48 x 48 | 0.06865 | 0.06625 | 49.37 | 10.30 | - |
| dvs_conv | 10 x 64 x 48 x 48 | 0.1535 | 0.02914 | 38.34 | 100.00 | - |
| dvs_bn | 10 x 64 x 48 x 48 | 1.077 | 1.065 | 45.93 | 100.00 | - |
| dvs_mem | 10 x 64 x 48 x 48 | 1.314 | 1.283 | 33.54 | 100.00 | - |
| dvs_spike | 10 x 64 x 48 x 48 | 0.2399 | 0.2236 | 51.32 | 5.76 | 5.76 |
| feature0_conv | 10 x 128 x 48 x 48 | 0.7965 | 0.685 | 27.93 | 100.00 | - |
| feature0_bn | 10 x 128 x 48 x 48 | 0.9794 | 0.9461 | 30.14 | 100.00 | - |
| feature0_mem | 10 x 128 x 48 x 48 | 1.223 | 1.149 | 24.15 | 100.00 | - |
| feature0_spike | 10 x 128 x 48 x 48 | 0.2607 | 0.2455 | 52.87 | 6.80 | 6.80 |
| pool1 | 10 x 128 x 24 x 24 | 0.1921 | 0.171 | 54.36 | 15.95 | - |
| feature2_conv | 10 x 256 x 24 x 24 | 1.419 | 1.039 | 29.88 | 100.00 | - |
| feature2_bn | 10 x 256 x 24 x 24 | 0.9751 | 0.9134 | 33.93 | 100.00 | - |
| feature2_mem | 10 x 256 x 24 x 24 | 1.373 | 1.24 | 31.28 | 100.00 | - |
| feature2_spike | 10 x 256 x 24 x 24 | 0.3081 | 0.2858 | 56.43 | 9.49 | 9.49 |
| feature3_conv | 10 x 256 x 24 x 24 | 3.915 | 2.728 | 25.13 | 100.00 | - |
| feature3_bn | 10 x 256 x 24 x 24 | 0.9766 | 0.9235 | 26.44 | 100.00 | - |
| feature3_mem | 10 x 256 x 24 x 24 | 1.387 | 1.264 | 25.58 | 100.00 | - |
| feature3_spike | 10 x 256 x 24 x 24 | 0.3403 | 0.3134 | 50.03 | 11.59 | 11.59 |
| pool2 | 10 x 256 x 12 x 12 | 0.2585 | 0.2219 | 62.03 | 25.10 | - |

Different spatial resolutions use the same normalized-radius convention,
so changes in fractions are descriptive and are not strict transfer gains.
Only retained-spatial tensors are analyzed; no FFT is applied after flattening.
