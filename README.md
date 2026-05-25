# Task-Aware Removed-TV Budget Certification

This code package accompanies the paper **Task-Aware Removed-TV Budget Certification for Multimedia Restoration Pipelines**. It contains the reproducibility code, CSV outputs, generated table snippets, and paper figures needed to inspect or rebuild the reported audit tables without shipping large datasets or model weights.

## Contents

- `methods/ectv.py`: removed-TV budget functions, TV energy, ECTV solver, and proposal certification wrapper.
- `experiments/run_tmm_pipeline.py`: main reproducibility pipeline for TV/classical/FDnCNN experiments and downstream tasks.
- `experiments/write_revision_audit_tables.py`: rebuilds revision/audit tables from existing CSV files.
- `experiments/run_drunet_subset.py`: DRUNet audit script; supports `--table-only`.
- `experiments/run_swinir_audit.py`: SwinIR audit script; supports `--table-only`.
- `results/`: reported CSV files and manifests.
- `generated/`: LaTeX table snippets generated from `results/`.
- `figures_tmm/`: final PDF figures used in the manuscript.
- `data/models/kair_minimal/` and `data/models/swinir_minimal/`: minimal network definitions used by optional learned-baseline audits.

Large raw datasets, processed image caches, virtual environments, PyTorch dependency folders, and model weights are intentionally excluded.

## Environment

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Optional learned-baseline audits need PyTorch and the corresponding public weights:

- `data/models/FDnCNN_color.mat`
- `data/models/drunet_color.pth`
- `data/models/swinir_color_noise25.pth`

These files are not included in the GitHub package.

## Rebuild Tables From Included CSVs

The included CSV files are enough to regenerate the paper tables:

```bash
python experiments/write_revision_audit_tables.py
python experiments/run_drunet_subset.py --table-only
python experiments/run_swinir_audit.py --table-only
```

## Run Experiments

A full image-level rerun requires Kodak24, PolyU real-noise pairs, and optional learned-model weights placed under `data/`.

```bash
python experiments/run_tmm_pipeline.py --max-iter 80
```

For a fast smoke test:

```bash
python experiments/run_tmm_pipeline.py --quick --max-iter 5
```

The pipeline writes CSV files to `results/`, table snippets to `generated/`, and figures to `figures_tmm/`.

## Scope

ECTV is an auditable admission/control layer for TV solvers and learned denoising proposals under a selected removed-TV budget. The released code supports the paper's task-aware certification and operating-region analysis; it is not packaged as a state-of-the-art denoising benchmark suite.
