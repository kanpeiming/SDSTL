# Stage 1 numeric digest

All rows are sample averages; band fractions omit zero-centered-energy samples.
Original RGB/Edge and DVS amplitudes have different meanings. This table is descriptive.

| Modality | Channel | Samples | Valid spectra | Low % | Mid % | High % | Activity % |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgb_gray_pixel | 0 | 100 | 100 | 84.91 | 12.87 | 2.21 | 99.96 |
| edge_dual_sobel | 0 | 100 | 100 | 43.60 | 34.97 | 21.43 | 100.00 |
| dvs_processed | 0 | 100 | 100 | 20.92 | 29.61 | 49.47 | 10.63 |
| dvs_processed | 1 | 100 | 100 | 19.10 | 27.98 | 52.92 | 9.46 |
| dvs_native_count | 0 | 100 | 100 | 64.55 | 20.52 | 14.93 | 8.10 |
| dvs_native_count | 1 | 100 | 100 | 58.69 | 22.05 | 19.26 | 7.04 |

## Input checks

- Edge channel correlation (valid samples): 1.000000
- DVS channel correlation (valid samples): 0.076042
- Samples with raw event files: 100/100
- Rebuilt vs saved DVS maximum absolute error: 0
- Raw events in last time bin, mean sample share: 0.001%

These measurements alone do not establish whether a frequency band helps classification.
