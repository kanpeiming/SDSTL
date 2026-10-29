# Stage 1 numeric digest

All rows are sample averages; band fractions omit zero-centered-energy samples.
Original RGB/Edge and DVS amplitudes have different meanings. This table is descriptive.

| Modality | Channel | Samples | Valid spectra | Low % | Mid % | High % | Activity % |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgb_gray_pixel | 0 | 20 | 20 | 84.86 | 13.07 | 2.07 | 100.00 |
| edge_dual_sobel | 0 | 20 | 20 | 44.18 | 34.05 | 21.77 | 100.00 |
| dvs_processed | 0 | 20 | 20 | 19.85 | 29.92 | 50.23 | 11.00 |
| dvs_processed | 1 | 20 | 20 | 18.21 | 28.27 | 53.53 | 9.74 |
| dvs_native_count | 0 | 20 | 20 | 64.03 | 21.21 | 14.76 | 8.40 |
| dvs_native_count | 1 | 20 | 20 | 58.43 | 22.69 | 18.87 | 7.26 |

## Input checks

- Edge channel correlation (valid samples): 1.000000
- DVS channel correlation (valid samples): 0.053200
- Samples with raw event files: 20/20
- Rebuilt vs saved DVS maximum absolute error: 0
- Raw events in last time bin, mean sample share: 0.001%

These measurements alone do not establish whether a frequency band helps classification.
