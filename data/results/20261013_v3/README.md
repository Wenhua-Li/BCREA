# Raw campaign records (not distributed)

This directory would hold the raw run records of the 20261013_v3 campaign
(5580 files, ~1.6 GB: `pools/*.npz`, `scans/*.json`, `reopt/*.npz`).
They are **not distributed** with this repository.

- Every paper table and 8 of the 11 figures are generated from the analysis layer in
  `data/results/` (the `*_analysis` CSV/JSON products) — those reproduce out of the box.
- Fig. 2, Fig. 5, and Fig. S5 read raw pools/scans; regenerate the campaign first:
  see README "Regenerating the raw campaign", or `docs/REPRODUCIBILITY.md`.
