# CEP-DVS transfer input audit

- Edge/DVS labelled files: 10000/10000
- Training pairs: 9000; valid DVS indices: 9000
- Same file ID: 9000/9000; same label: 9000/9000
- Invalid DVS test indices: 0
- Edge samples missing from DVS: 0; DVS samples missing from Edge: 0
- Reference samples with offline Edge: 100/100
- Stored offline Edge label mismatches among loaded samples: 0
- Dataloader default pathFile.csv: /home/user/Datasets/kpm-own/CEP-DVS/pathFile.csv; exists=True; matches audited CSV=True

- Offline Edge channel 0: mean high-band 21.86%; raw active pixels 99.45%
- Offline Edge channel 1: mean high-band 22.89%; raw active pixels 99.75%

This reproduces the current loader's index rules. It does not train or evaluate a model.
