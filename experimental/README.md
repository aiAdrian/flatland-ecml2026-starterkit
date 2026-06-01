# Experimental Setup Overview

This folder contains the standalone migration and reference material for the new Flatland workflow.

## Goal

Provide a clean migration workspace for Flatland experiments with:
- Baselines: `random`, `deadlock avoidance (DLA)`
- Training stages: `behavior cloning (BC)`, then `MAPPO`
- Optional live rendering during eval

## Current Status

- Research/reference code is available under `experimental/flatland-baselines/` and
  `experimental/flatland_minimal_project_NOT_WORKING_TRANSFORM/`.
- New standalone runner exists under `experimental/flatland_solver/` with
  policy-centered layout (`policy/dla`, `policy/random`, `policy/mappo`).

## Folder Structure (experimental)

- `experimental/examples/`
  - Small standalone examples (for quick local validation)
- `experimental/flatland-baselines/`
  - Upstream baseline reference code (including DLA implementation)
- `experimental/flatland_minimal_project_NOT_WORKING_TRANSFORM/`
  - Legacy/minimal MARL project used for feature and architecture extraction
- `experimental/requirements_experimental_latest.txt`
  - Consolidated dependency set for experimental migration
- `experimental/utils/flatland_env_persist_util.py`
  - Static helper for Flatland environment save/load/generation via pickle
- `experimental/flatland_solver/`
  - Main runnable structure (main entry point + policy-centric modules)

## Quick Start

From repository root:

```bash
cd /home/u216993/workspace/ai4realnet/flatland-ecml2026-starterkit

# activate pyenv env
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"
# pyenv activate flatland-ecml2026

# install/update deps
pip install -r experimental/requirements_experimental_latest.txt

# inspect available experimental sources
find experimental -maxdepth 3 -type d | sort

# run new policy-centered solver
cd experimental/flatland_solver
python main.py --mode eval --policy random --episodes 3
python main.py --mode eval --policy dla --episodes 3
python main.py --mode eval --policy dla --episodes 1 --rendering
```

## Planned Build Order

1. Add BC data collection, training, and checkpoint loading.
2. Add MAPPO training and checkpoint loading.
3. Add eval for trained `mappo` (checkpoint loading).
4. Add observation variants and comparisons.

## Notes

- `experimental/flatland-baselines/` and `experimental/flatland_minimal_project_NOT_WORKING_TRANSFORM/` are intentionally ignored by git in this repo.
- Rendering is live-window based, no mandatory video export.
