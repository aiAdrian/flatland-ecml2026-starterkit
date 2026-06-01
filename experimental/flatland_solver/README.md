# flatland_solver (Policy-Centered Structure)

This structure is organized by policy to keep related code together.

## Layout

- `main.py`
- `utils/`
  - `env_factory.py`
  - `action_utils.py`
- `policy/random/`
  - `policy.py`
  - `observation.py`
- `policy/dla/`
  - `policy.py`
  - `observation.py`
- `policy/mappo/`
  - `policy.py`
  - `observation.py`
- `policy/bc/`
  - `policy.py`
  - `observation.py`
  - `trainer.py`

## Install  
 
### pyenv 
```bash
pyenv install 3.12.11
pyenv virtualenv 3.12.11 flatland-ecml2026
pyenv activate flatland-ecml2026
 
pip install -r requirements_experimental_latest.txt
```

## pyenv
```bash
pyenv activate flatland-ecml2026 
```

## Run

```bash
cd experimental/flatland_solver

python main.py --mode eval --policy random --episodes 3
python main.py --mode eval --policy dla --episodes 3
python main.py --mode eval --policy dla --episodes 1 --rendering

# prepare reusable PKL scenarios for faster repeated training/eval
python main.py --prepare-pkls --prepare-only --pkl-dir pkl_envs --pkl-count 64 --pkl-seed-start 1000

# curriculum-style cache (legacy behavior): N envs per agent-count
python main.py --prepare-pkls --prepare-only --pkl-dir pkl_envs \
  --agent-curriculum 1 2 3 4 5 7 10 15 20 --pkl-num-envs-per-agent 5

# train BC and evaluate from checkpoint
python main.py --mode train --policy bc --episodes 2 --train-epochs 1
python main.py --mode eval --policy bc --episodes 2

# same with PKL-backed environments + debug checks
python main.py --mode train --policy bc --env-source pkl --pkl-dir pkl_envs --episodes 3 --train-epochs 2 --debug-checks

# train MAPPO and evaluate from checkpoint
python main.py --mode train --policy mappo --episodes 2 --train-epochs 1
python main.py --mode eval --policy mappo --episodes 2

# MAPPO diagnostics from legacy doc guidance (entropy / approx_kl)
python main.py --mode train --policy mappo --env-source pkl --pkl-dir pkl_envs --episodes 3 --train-epochs 2 --debug-checks

# tensorboard (runs include mode/policy/params in folder name)
tensorboard --logdir runs

# legacy observation variants from flatland_minimal_project_NOT_WORKING_TRANSFORM
python main.py --mode train --policy bc --obs-variant decision_point --episodes 2 --train-epochs 1
python main.py --mode train --policy mappo --obs-variant spawn_aware --episodes 2 --train-epochs 1
python main.py --mode eval --policy mappo --obs-variant conflict_aware --episodes 1

# extract KPI hints from legacy PDF
python tools/pdf_kpi_digest.py \
  --pdf ../flatland_minimal_project_NOT_WORKING_TRANSFORM/2026_05_FLATLAND_MARL_EXPERIMENT.pdf \
  --out-md runs/legacy_kpi_digest.md \
  --out-txt runs/legacy_kpi_raw.txt
```

Each run creates a folder like:

`runs/YYYYMMDD_HHMMSS_<mode>_<policy>_a<n_agents>_w<width>h<height>_c<n_cities>_s<seed>`

Logged metrics include:

- eval: `done_rate`, `episode_len`, `total_reward`, `deadlock_rate`
- eval summary: `success_rate`, `avg_steps`, `avg_reward`, `avg_deadlock_rate`
- bc: `loss`, `accuracy`
- mappo/ppo: `p_loss`, `v_loss`, `entropy`, `approx_kl`

Observation variants (`--obs-variant`) for BC/MAPPO:

- `fast_tree`: existing 36+mask observation (`reinforcement-learning/my_observation_builder.py`)
- `decision_point`: legacy decision-point observation (22D base)
- `spawn_aware`: legacy spawn-aware observation (25D base)
- `conflict_aware`: legacy conflict-aware observation (44D base)

Models now infer and store `obs_dim` in checkpoints, so BC/MAPPO can train/eval across these variants.
On Flatland 4.2.5, `conflict_aware` may print repeated DLA helper warnings from legacy internals; training/eval still proceeds.

The legacy PDF in `../flatland_minimal_project_NOT_WORKING_TRANSFORM/`
is treated as guidance for diagnostics and KPI tracking. The utility script above
creates a searchable digest so KPI terms and baseline claims are visible in one place.

## 15-20 Min Validation Flow

Run this block to verify PKL generation, BC/MAPPO train+eval, and KPI artifacts:

```bash
cd experimental/flatland_solver

# 1) reusable PKL scenarios
python main.py --prepare-pkls --prepare-only --pkl-dir pkl_envs --pkl-count 32 --pkl-seed-start 1000

# 2) BC train + eval on PKL envs
python main.py --mode train --policy bc --env-source pkl --pkl-dir pkl_envs --episodes 3 --train-epochs 2 --max-episode-steps 60 --debug-checks
python main.py --mode eval --policy bc --env-source pkl --pkl-dir pkl_envs --episodes 3 --max-episode-steps 60

# 3) MAPPO train + eval on PKL envs
python main.py --mode train --policy mappo --env-source pkl --pkl-dir pkl_envs --episodes 3 --train-epochs 2 --max-episode-steps 60 --debug-checks
python main.py --mode eval --policy mappo --env-source pkl --pkl-dir pkl_envs --episodes 3 --max-episode-steps 60

# 4) tensorboard
tensorboard --logdir runs
```

## Expected Debug Output

- BC (`--debug-checks`): `mask_rows_fixed`, `labels_forced_valid`
- MAPPO (`--debug-checks`): `entropy`, `approx_kl`
- Eval summary: `success_rate`, `avg_steps`, `env_source`

## PKL Dataset Generation: Controlling Agent Count

PKL environments are pre-generated rail grids cached for fast train/eval cycles. You can control how many agents each environment has:

### Single Agent Count (Fixed)

Generate 1000 environments with a fixed agent count:

```bash
cd experimental/flatland_solver

# 1000 PKLs with 5 agents (default)
python main.py --prepare-pkls --prepare-only \
  --pkl-count 1000

# 500 PKLs with 10 agents
python main.py --prepare-pkls --prepare-only \
  --pkl-count 500 --n-agents 10

# Clear all generated PKLs and checkpoints (start fresh)
rm -rf pkl_envs checkpoints

# Variable Agent-Counts: 25 PKLs per agent-count (8 counts = 200 total)
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 1 2 3 4 5 7 10 20 \
  --pkl-num-envs-per-agent 25
```

### Agent Curriculum (Variable)

Generate environments with **multiple agent counts** to train robust policies across agent scales:

```bash
# 100 PKLs for each of 5 agent-counts = 500 total
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 2 5 10 15 20 \
  --pkl-num-envs-per-agent 100

# 50 PKLs for each of 6 agent-counts = 300 total
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 1 3 5 7 10 15 \
  --pkl-num-envs-per-agent 50
```

Both approaches generate environments in `pkl_envs/` and are used identically during training/evaluation:

```bash
# Train on ANY mix of pre-generated PKLs
python main.py --mode train --policy bc --env-source pkl \
  --pkl-dir pkl_envs --episodes 200 --train-epochs 5

# Eval on ANY mix of pre-generated PKLs
python main.py --mode eval --policy bc --env-source pkl \
  --pkl-dir pkl_envs --episodes 20
```

The agent curriculum approach creates robust policies that generalize across agent counts without needing curriculum sampling during training—all agent-counts are mixed in each epoch.

## DLA Record -> Offline BC -> MAPPO Warmstart

Use this workflow to run DLA only once, then train BC for many epochs on a saved dataset, and finally warmstart MAPPO from the BC checkpoint.

```bash
cd experimental/flatland_solver

# Optional: clear generated caches and models
rm -rf pkl_envs checkpoints datasets runs

# 1) Build PKL cache (example: variable agent counts)
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 1 2 3 4 5 7 10 20 \
  --pkl-num-envs-per-agent 25

# 2) Record DLA dataset once (writes datasets/dla_dataset.pt)
python main.py --mode record --policy bc \
  --env-source pkl --pkl-dir pkl_envs \
  --episodes 200 --max-episode-steps 300 \
  --obs-variant decision_point \
  --dataset-path datasets/dla_dataset.pt

# 3) Offline BC optimization for many epochs (fast, no DLA at train time)
python main.py --mode train --policy bc \
  --dataset-path datasets/dla_dataset.pt \
  --train-epochs 20 --batch-size 256 --lr 3e-4 \
  --bc-checkpoint checkpoints/bc.pt

# 4) MAPPO warmstart from BC checkpoint
python main.py --mode train --policy mappo \
  --env-source pkl --pkl-dir pkl_envs \
  --episodes 200 --train-epochs 5 \
  --obs-variant spawn_aware \
  --init-checkpoint checkpoints/bc.pt \
  --mappo-checkpoint checkpoints/mappo.pt
```

Notes:

- `--mode record` saves `(features, action-mask, DLA-action)` samples to `--dataset-path`.
- `--mode train --policy bc` uses offline mode automatically when `--dataset-path` exists.
- `--init-checkpoint` is used by MAPPO to warmstart model weights from BC.
- Keep `--obs-variant` consistent between recording and BC training for best results.

## KPI Artifacts

- TensorBoard logs: `runs/`
- PDF KPI digest: `runs/legacy_kpi_digest.md`
- Extracted raw PDF text: `runs/legacy_kpi_raw.txt`

## Complete Reset & Automated Test Suite

A shell script is provided to clean up all previous runs and generated data, then execute the full test pipeline in sequence:

```bash
# From inside experimental/flatland_solver:
./run_flatland_full_reset_and_test.sh
```

**What the script does:**
1. Removes `runs/`, `pkl_envs/`, and `checkpoints/` directories to start fresh
2. Runs random policy eval (baseline sanity check)
3. Runs DLA policy eval (legacy heuristic baseline)
4. Trains BC for 200 episodes × 5 epochs on `decision_point` observations
5. Evaluates BC on 20 episodes
6. Trains MAPPO for 200 episodes × 5 epochs on `spawn_aware` observations
7. Evaluates MAPPO on 20 episodes

All output (progress bars, metrics, TensorBoard logs) is displayed live in the console. This typically takes **2–4 hours** depending on your hardware.

If you prefer a faster smoke test, run the **15-20 Min Validation Flow** above instead.

## Troubleshooting

- If you use `--env-source pkl` and get "No PKL environments found", run `--prepare-pkls --prepare-only` first.
- Flatland may print "Line Generator should not have random state." This warning is non-fatal for these workflows.
- If `tensorboard` fails with `ModuleNotFoundError: No module named 'pkg_resources'`, install a compatible setuptools version: `python -m pip install "setuptools<81"`.
- BC uses DLA as expert; DLA is reinitialized per episode to avoid stale internal maps when cycling through many PKL files.
