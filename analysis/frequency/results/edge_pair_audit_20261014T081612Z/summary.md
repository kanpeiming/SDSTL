# CEP-DVS transfer input audit

- Edge/DVS labelled files: 10000/10000
- Training pairs: 9000; valid DVS indices: 9000
- Same file ID: 9000/9000; same label: 9000/9000
- Invalid DVS test indices: 0
- Edge samples missing from DVS: 0; DVS samples missing from Edge: 0
- Reference samples with offline Edge: 20/20
- Stored offline Edge label mismatches among loaded samples: 0
- Dataloader default pathFile.csv: /home/user/kpm/kpm/Dataset/CEP-DVS/pathFile.csv; exists=False; matches audited CSV=False

- Offline Edge channel 0: mean high-band 21.18%; raw active pixels 99.20%
- Offline Edge channel 1: mean high-band 23.93%; raw active pixels 99.20%

WARNING: The default training dataloader CSV is missing or differs. Pair results use --dataset-root/pathFile.csv and may not match an unmodified training run.

This reproduces the current loader's index rules. It does not train or evaluate a model.
