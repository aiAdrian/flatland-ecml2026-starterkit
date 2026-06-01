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

## KPI Artifacts

- TensorBoard logs: `runs/`
- PDF KPI digest: `runs/legacy_kpi_digest.md`
- Extracted raw PDF text: `runs/legacy_kpi_raw.txt`

## Troubleshooting

- If you use `--env-source pkl` and get "No PKL environments found", run `--prepare-pkls --prepare-only` first.
- Flatland may print "Line Generator should not have random state." This warning is non-fatal for these workflows.
- BC uses DLA as expert; DLA is reinitialized per episode to avoid stale internal maps when cycling through many PKL files.
