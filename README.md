# BCREA: Certified Engagement-Plan Menus for Air-Defense SWTA under Budgeted Capability Degradation

Code and analysis data for the paper *"Certified Engagement-Plan Menus for Air-Defense
SWTA under Budgeted Capability Degradation"* (IEEE TAES submission).

BCREA (Band-confined Coverage-driven Robustness-aware Evolutionary Algorithm) is a
two-stage evolutionary algorithm designed around the certification objective:
band-confined robustness-aware environmental selection populates the pool **inside the
nominal equivalence band**, and coverage-based extraction then returns a small certified
plan menu by greedy robust-coverage selection, under a budgeted uncertainty set
U(Gamma, delta) over 12 capability-degrading components (6 radar detection ranges, 6
launcher engagement boundaries). The variation operators and the decoder follow the
knowledge-guided SWTA literature (KCEA — cited as inspiration, experimental baseline, and
benchmark source); the band-confined selection and the coverage-based extraction are new
components. The paper reports set-level robustness metrics — worst-case degradation
Delta, robust coverage RC, and the robust covering number m* — with m* computed exactly
(MILP via SciPy/HiGHS).

## Repository layout

| Directory | Contents |
|---|---|
| `core/` | SWTA model, constructive decoder (CPU + optional fused CUDA kernel), KCEA baseline engine, scenario I/O, robustness metrics (`robust_metrics.py`) |
| `experiments/` | Method entry points (`run_*.py`), scenario generator, task-list generator (`make_tasks.py`), parallel local executor (`local_run.py`) |
| `analysis/` | Batch analysis (`analyze_v3*.py`) and paper-table generation (`make_tables_v3.py`) |
| `figures/` | One script per paper figure (shared `style.py` / `io.py`) |
| `tests/` | Protocol, baseline-regression, and GPU-acceptance unit tests |
| `data/scenarios/` | 12 TEVC benchmark instances + 18 generated instances (with manifests and generation seeds) |
| `data/results/` | Analysis-layer CSV/JSON of the 5580-run campaign — the direct data source of every paper table and figure |
| `data/tasks/` | Executable task lists (seeds and outputs explicit) + per-run manifest |
| `docs/` | `REPRODUCIBILITY.md` (campaign composition, seeds, instance-label map), per-run drill table |

`v3` in file names refers to the locked experimental protocol version of the reported
campaign (batch `20261013_v3`), not to a software release number.

## Installation

Python 3.11 with:

```
numpy  scipy  pandas  matplotlib
```

(`cupy-cuda12x` is optional: `core/decoder_gpu.py` provides a fused CUDA batch decoder
that is bit-identical to the CPU decoder and only changes speed. Everything runs on CPU
by default.)

```
pip install -r requirements.txt
```

## Quick checks

From the repository root:

```
PYTHONPATH=. python tests/test_v3_protocol.py     # 7/7 expected
PYTHONPATH=. python tests/test_baselines.py       # 3/3 expected
PYTHONPATH=. python tests/test_gpu_backend.py     # GPU parts skip without a CUDA device
```

## Reproducing the paper's tables and figures

All tables and 8 of the 11 figures are generated **from the distributed analysis layer
alone** (`data/results/`), with no raw run records needed:

```
# all paper tables -> paper/tables/*.tex (+ paper/tables_v3.md, docs/tab_drill_per_run.tex)
PYTHONPATH=. python analysis/make_tables_v3.py

# figures that read only the analysis layer
python figures/fig4_comparison.py      # Fig. 4  six-method comparison
python figures/fig6_mechanism.py       # Fig. 6  mechanism / mismatch boundary
python figures/fig7_switching.py       # Fig. 7  switch drill and latency
python figures/fig8_ident.py           # Fig. 8  switching under imperfect identification
python figures/figS1_grid.py           # Fig. S1
python figures/figS2_mstar.py          # Fig. S2
python figures/figS3_conv.py           # Fig. S3
python figures/figS4_mismatch.py       # Fig. S4
```

PDF outputs land in `paper/figures/`. Three figures (Fig. 2, Fig. 5, Fig. S5) read raw
pools/scans and therefore require the raw campaign (below).

## Regenerating the raw campaign (optional)

The raw run records (~1.6 GB: 1800 pool `.npz`, 1800 scan `.json`, 1980 reopt `.npz`)
are not distributed here. To regenerate them end-to-end:

```
PYTHONPATH=. python experiments/make_tasks.py        # writes data/tasks/*
python experiments/local_run.py --taskfile data/tasks/tasks_20261013_v3_pools.txt  --out data/results/20261013_v3/pools
python experiments/local_run.py --taskfile data/tasks/tasks_20261013_v3_scans.txt  --out data/results/20261013_v3/scans
python experiments/local_run.py --taskfile data/tasks/tasks_20261013_v3_reopt_5seed.txt --out data/results/20261013_v3/reopt
PYTHONPATH=. python analysis/analyze_v3.py           # rebuilds the analysis layer
```

Seed values, budget conventions, and the instance-label map are documented in
`docs/REPRODUCIBILITY.md`.

## Reproducing one pool / one scan / one re-optimization

```
PYTHONPATH=. python core/kcea.py --scenario data/scenarios/tevc/Info_enemy_1.mat --seed 1 \
    --fes 9000x --selection matlab --out /tmp/nominal_i1_s01.npz
PYTHONPATH=. python experiments/run_pool_scan.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
    --pool /tmp/nominal_i1_s01.npz --deltas 0.3 --gammas 2 --n-scen 60 --scen-seed 6000 --out /tmp/scan_i1.json
PYTHONPATH=. python experiments/run_corner_pool.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
    --components 0,1 --seed 1 --fes 9000x --out /tmp/reopt_i1_c0c1.npz
```

## Baselines

- `experiments/run_nsga2_pool.py` — bi-objective NSGA-II (nominal objective, worst-case
  degradation), including the elitism/dominance fixes documented in the paper.
- `experiments/run_sampled_wc.py` — sampled worst-case optimization (M=16 stratified
  realizations, seed 9101).
- `experiments/run_pessimistic_pool.py` — pessimistic re-solve baseline pool.
- `core/kcea.py` — the nominal single-solution engine, run entry for the nominal method.
- `tests/test_baselines.py` reproduces the baseline regression checks reported in the
  paper (3/3 expected).

## Budget convention

The paper convention is a common true-decode budget D = 9000·N per run; a scored BCREA
individual costs (K+1) = 9 decodes (K = 8), and the sampled worst-case optimizer costs
(1+M) = 17. Details in `docs/REPRODUCIBILITY.md`.

## License

MIT — see `LICENSE`.
