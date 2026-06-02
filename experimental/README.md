# Experimental Workspace

This directory contains the maintained Flatland experiment runner plus older reference material that was used during migration.

## What Is Active

The main runnable path is `experimental/flatland_solver/`.

It supports:
- Baseline evaluation: `random`, `dla`
- Dataset recording from DLA
- Offline behavior cloning (BC)
- MAPPO training with optional BC warmstart
- Shared observation factory for BC and MAPPO
- PKL-backed environment caches in `generated_envs/`
- Outcome-based reward shaping
- TensorBoard logging and compact console output

The detailed command reference lives in `experimental/flatland_solver/README.md`.

## What Is Reference Only

These directories are retained as migration/reference material and are not required as runtime dependencies for `experimental/flatland_solver/`:
- `experimental/flatland-baselines/`
- `experimental/flatland_minimal_project_NOT_WORKING_TRANSFORM/`

They are useful for comparison, provenance, or one-off lookup, but the solver now carries the needed BC/MAPPO observation code and DLA runtime code locally.

## Folder Overview

- `experimental/flatland_solver/`
  Main runnable experiment stack.
- `experimental/examples/`
  Small standalone examples and scratch material.
- `experimental/flatland-baselines/`
  Upstream/reference baseline sources.
- `experimental/flatland_minimal_project_NOT_WORKING_TRANSFORM/`
  Older MARL reference project used during extraction/migration.
- `experimental/requirements_experimental_latest.txt`
  Experimental dependency set.
- `experimental/utils/flatland_env_persist_util.py`
  Helper for Flatland environment persistence/generation.

## Quick Start

From repository root:

```bash
cd experimental/flatland_solver

# install/update experimental deps from repository root beforehand if needed:
# pip install -r experimental/requirements_experimental_latest.txt

# baseline checks
python main.py --mode eval --policy random --episodes 3
python main.py --mode eval --policy dla --episodes 3

# prepare reusable PKL environments
python main.py --prepare-pkls --prepare-only --pkl-dir generated_envs --pkl-count 32 --pkl-seed-start 1000

# record DLA dataset, train BC, then train MAPPO
python main.py --mode record --policy bc --env-source pkl --pkl-dir generated_envs --episodes 20 --obs-variant decision_point
python main.py --mode train --policy bc --dataset-path datasets/dla_dataset.pt --train-epochs 5 --obs-variant decision_point
python main.py --mode train --policy mappo --env-source pkl --pkl-dir generated_envs --init-checkpoint checkpoints/bc.pt --episodes 20 --obs-variant spawn_aware
```

## Auxiliary Tools

- `experimental/flatland_solver/run_flatland_full_reset_and_test.sh`
  Full reset plus baseline/BC/MAPPO smoke-to-long run script.
- `experimental/flatland_solver/tools/pdf_kpi_digest.py`
  Utility to extract KPI-related text from PDFs into searchable artifacts.

## Notes

- `experimental/flatland_solver/` is the maintained execution path.
- The Starter Kit root and `reinforcement-learning/` are kept separate from solver-local refactors.
- Rendering is live-window based; no mandatory video export pipeline is assumed here.
