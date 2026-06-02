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

# MAPPO with curriculum (repeat mode: same agent count)
python main.py --mode train --policy mappo --curriculum-spec 5 --curriculum-repeat 10 --episodes 10 --train-epochs 1

# MAPPO with curriculum (sequence mode: varying agent counts)
python main.py --mode train --policy mappo --curriculum-spec 1x5,5x5 --episodes 10 --train-epochs 1

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

## Curriculum Selection: Repeat vs. Sequence Mode

Training can use **agent curriculum** to expose policies to varying agent counts. Two modes are supported:

### Repeat Mode (Single Agent Count)

Repeat a fixed agent count across training episodes:

```bash
# Train with 5-agent environments only, repeated 10 times
python main.py --mode train --policy mappo \
  --curriculum-spec 5 --curriculum-repeat 10 \
  --episodes 100

# Shorthand: generates [5]*10 curriculum
```

Use this for **homogeneous** training (all episodes have same complexity).

### Sequence Mode (Variable Agent Counts)

Cycle through multiple agent counts in a controlled order:

```bash
# Curriculum: 10 episodes with 3 agents, then 10 with 5 agents, then 5 with 7 agents
python main.py --mode train --policy mappo \
  --curriculum-spec 3x10,5x10,7x5 \
  --episodes 25

# Generates [3]*10 + [5]*10 + [7]*5
```

Use this for **curriculum learning** (policy learns from simple→complex scenarios).

### Curriculum Selection with PKL Datasets

When combining with `--env-source pkl`:

```bash
# Generate variable-agent PKL cache
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 1 2 3 4 5 7 10 20 \
  --pkl-num-envs-per-agent 25

# Train with repeat mode (all PKLs, randomly selected)
python main.py --mode train --policy mappo \
  --env-source pkl --pkl-dir pkl_envs \
  --curriculum-spec 5 --curriculum-repeat 20 \
  --episodes 100

# Train with sequence mode (curriculum marker printed at transitions)
python main.py --mode train --policy mappo \
  --env-source pkl --pkl-dir pkl_envs \
  --curriculum-spec 1x10,3x10,5x10,10x10 \
  --episodes 40
```

During sequence-mode training, curriculum transitions are marked in output:

```
[CURRICULUM] n_agents=1 @ episode 1/40
[TRAIN] ep=1/40 (2%) done=50% reward=+15.30 ...
...
[CURRICULUM] n_agents=3 @ episode 11/40
[TRAIN] ep=11/40 (27%) done=75% reward=+25.10 ...
```

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

### Full Training (Legacy-Close MAPPO Loop)

This is the recommended end-to-end run when you want legacy-like MAPPO behavior
(collect rollout episodes first, then PPO K-epoch mini-batch updates, with
mid-training eval checkpoints).

```bash
cd experimental/flatland_solver

# 0) clean start
rm -rf pkl_envs checkpoints datasets runs

# 1) generate PKL curriculum cache (+ metadata index pkl_envs/pkl_index.json)
python main.py --prepare-pkls --prepare-only \
  --agent-curriculum 1 2 3 4 5 7 10 20 \
  --pkl-num-envs-per-agent 25

# 2) record DLA demonstrations once
python main.py --mode record --policy bc \
  --env-source pkl --pkl-dir pkl_envs \
  --episodes 200 --max-episode-steps 300 \
  --obs-variant decision_point \
  --dataset-path datasets/dla_dataset.pt

# 3) offline BC optimization
python main.py --mode train --policy bc \
  --dataset-path datasets/dla_dataset.pt \
  --train-epochs 20 --batch-size 256 --lr 3e-4 \
  --bc-checkpoint checkpoints/bc.pt

# 4) MAPPO warmstart + legacy-close train loop
python main.py --mode train --policy mappo \
  --env-source pkl --pkl-dir pkl_envs \
  --episodes 200 --train-epochs 5 --max-episode-steps 300 \
  --obs-variant spawn_aware \
  --init-checkpoint checkpoints/bc.pt \
  --mappo-checkpoint checkpoints/mappo.pt \
  --mappo-rollout-episodes 10 \
  --mappo-ppo-epochs 4 \
  --mappo-batch-size 256 \
  --mappo-entropy-coef 0.02 \
  --mappo-value-coef 0.5 \
  --mappo-clip-eps 0.2 \
  --mappo-target-kl 0.05 \
  --mappo-kl-stop-factor 1.5 \
  --mappo-done-window 50 \
  --mappo-mid-eval-every 50 \
  --mappo-mid-eval-episodes 10 \
  --mappo-eval-greedy
```

After training:

```bash
# evaluate final checkpoints
python main.py --mode eval --policy bc --env-source pkl --pkl-dir pkl_envs --episodes 20 --obs-variant decision_point
python main.py --mode eval --policy mappo --env-source pkl --pkl-dir pkl_envs --episodes 20 --obs-variant spawn_aware

# inspect TensorBoard
tensorboard --logdir runs
```

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

## Legacy Migration Pipeline (Auto PKL + Optional BC Warmstart)

For the exact flow you described (generate scenarios if missing, DLA demo collect,
BC train from demos, optional BC warmstart into MAPPO, checkpoints + eval, full
console and TensorBoard logging), use:

```bash
cd experimental/flatland_solver

# Default: with BC warmstart
./run_legacy_migration_pipeline.sh

# MAPPO-only path (no BC record/train, no BC warmstart)
./run_legacy_migration_pipeline.sh --without-bc

# Legacy observation perf/debug reports (ObsFnPerf, feature reports)
LEGACY_OBS_DEBUG=1 LEGACY_OBS_SEARCH_DEPTH=4 ./run_legacy_migration_pipeline.sh
```

Default generated scenario directory is now `generated_envs/` (old `pkl_envs/`
still works when passed explicitly via `--pkl-dir`).

What this script does:

1. Builds PKL scenarios only when missing in `generated_envs/` (or forced with `--force-regenerate-pkls`).
2. Records DLA demonstrations (`--mode record`) when BC is enabled.
3. Trains BC offline from the recorded dataset (`--mode train --policy bc`).
4. Starts MAPPO training with optional `--init-checkpoint` from BC.
5. Saves MAPPO checkpoint and runs eval (and BC eval when BC is enabled).
6. Logs everything to console + TensorBoard (`runs/`).

You can override key run lengths via environment variables, for example:

```bash
RECORD_EPISODES=100 BC_EPOCHS=10 MAPPO_EPISODES=120 ./run_legacy_migration_pipeline.sh
```

## Troubleshooting

- If you use `--env-source pkl` and get "No PKL environments found", run `--prepare-pkls --prepare-only` first.
- Flatland may print "Line Generator should not have random state." This warning is non-fatal for these workflows.
- If `tensorboard` fails with `ModuleNotFoundError: No module named 'pkg_resources'`, install a compatible setuptools version: `python -m pip install "setuptools<81"`.
- BC uses DLA as expert; DLA is reinitialized per episode to avoid stale internal maps when cycling through many PKL files.

## Output Format & Logging

Training and evaluation outputs are designed for readability and debugging:

### Episode Format

Each episode produces a **single-line output (≤210 characters)**:

```
[TRAIN] ep=42/100 (42%) done=100% reward=+55.20 loss=0.234 kl=0.015 ent=1.523 actions=[pick=120 move=85 ...] samples=1024 steps=45
[BC] ep=5/10 (50%) done=75% reward=+30.10 loss=0.089 samples=256 steps=23
[REC] ep=15/200 (7%) done=50% reward=+10.50 loss=0.156 actions=[...] steps=60
[EVAL] ep=1/20 (5%) done=100% reward=+50.00 steps=42
```

Metrics included:
- `ep=X/Y (P%)`: Episode number and percentage complete
- `done=%`: Completion rate (% of agents that reached destination)
- `reward=±X.XX`: Total reward (shaped per episode)
- `loss=X.XXX`: Policy loss (BC/MAPPO)
- `kl=X.XXX`: KL divergence (MAPPO only)
- `ent=X.XXX`: Entropy (MAPPO only)
- `actions=[...]`: Action histogram (top 3 actions)
- `samples=N`: Batch samples collected
- `steps=N`: Episode length

### PPO Update Format

MAPPO training prints one line per **PPO update batch**:

```
[PPO] ep=1 (1/1) batch_1/4 samples=1024 kl=+0.018 ratio=1.05 p_loss=0.234 v_loss=0.089 clip=0.12 ent=1.52
[PPO] ep=1 (1/1) batch_2/4 samples=1024 kl=+0.022 ratio=1.04 p_loss=0.210 v_loss=0.095 clip=0.08 ent=1.51
[PPO] ep=1 (1/1) batch_3/4 samples=1024 kl=-0.003 ratio=0.98 p_loss=0.189 v_loss=0.082 clip=0.02 ent=1.50
[PPO] ep=1 (1/1) batch_4/4 samples=1024 kl=+0.015 ratio=1.02 p_loss=0.201 v_loss=0.088 clip=0.05 ent=1.49
```

Metrics:
- `ep=X (N/T)`: Epoch and update block index
- `batch_N/T`: Minibatch within PPO epoch
- `kl=±X.XXX`: Approximate KL divergence
- `ratio=X.XX`: Mean importance-sampling ratio
- `p_loss=X.XXX`: Policy loss
- `v_loss=X.XXX`: Value loss
- `clip=X.XX`: Clipping fraction
- `ent=X.XX`: Entropy

### Curriculum Markers

When using sequence-mode curriculum, transitions between agent counts are marked:

```
[CURRICULUM] n_agents=1 @ episode 1/100
[CURRICULUM] n_agents=5 @ episode 11/100
[CURRICULUM] n_agents=10 @ episode 21/100
```

## Output Metrics in TensorBoard & Console

Standard metrics logged to TensorBoard:

- **TRAIN**: `episode/reward`, `episode/done_rate`, `episode/steps`, `ppo/p_loss`, `ppo/v_loss`, `ppo/entropy`, `ppo/approx_kl`
- **EVAL**: `eval/done_rate`, `eval/avg_reward`, `eval/avg_steps`, `eval/deadlock_rate`
- **BC**: `bc/loss`, `bc/accuracy`

## CLI Reference: Curriculum & Training Flags

### Curriculum Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--curriculum-spec` | `str` | `None` | Curriculum spec: `"5"` (repeat) or `"3x10,5x10"` (sequence) |
| `--curriculum-mode` | `{auto,repeat,sequence}` | `auto` | How to parse `--curriculum-spec` |
| `--curriculum-repeat` | `int` | `None` | Repeat count for repeat mode (e.g., `5` + repeat 10 → [5]*10) |
| `--agent-curriculum` | `int [int ...]` | `None` | Legacy: list agent counts directly (e.g., `--agent-curriculum 1 2 3 5`) |

### MAPPO Training Control

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mappo-rollout-episodes` | `int` | `10` | Episodes to collect before PPO update block |
| `--mappo-ppo-epochs` | `int` | `4` | PPO epochs per update block |
| `--mappo-batch-size` | `int` | `256` | Mini-batch size for PPO updates |
| `--mappo-entropy-coef` | `float` | `0.02` | Entropy regularization coefficient |
| `--mappo-value-coef` | `float` | `0.5` | Value loss coefficient |
| `--mappo-clip-eps` | `float` | `0.2` | PPO clipping epsilon |
| `--mappo-target-kl` | `float` | `0.05` | Target KL for early-stop guard |
| `--mappo-kl-stop-factor` | `float` | `1.5` | Early-stop threshold multiplier |
| `--mappo-mid-eval-every` | `int` | `0` | Run mid-training eval every N collected episodes |
| `--mappo-mid-eval-episodes` | `int` | `10` | Episodes per mid/final eval run |

### Environment & Observation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--obs-variant` | `{fast_tree,decision_point,spawn_aware,conflict_aware}` | `fast_tree` | Observation variant for policies |
| `--obs` | (alias) | — | Legacy alias for `--obs-variant` |
| `--env-source` | `{generated,pkl}` | `generated` | Environment source (on-the-fly or pre-cached) |
| `--pkl-dir` | `Path` | `generated_envs` | Directory for PKL cache |
| `--pkl-count` | `int` | `32` | Number of PKL environments to generate |
| `--max-episode-steps` | `int` | `300` | Max steps per episode |

### Reward Shaping

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--disable-outcome-reward` | `flag` | `False` | Disable legacy outcome-based reward shaping |

Example with all curriculum + MAPPO settings:

```bash
python main.py --mode train --policy mappo \
  --env-source pkl --pkl-dir pkl_envs \
  --curriculum-spec 1x10,3x20,5x30 \
  --episodes 60 --train-epochs 3 \
  --mappo-rollout-episodes 10 \
  --mappo-ppo-epochs 4 \
  --mappo-batch-size 256 \
  --mappo-entropy-coef 0.02 \
  --mappo-value-coef 0.5 \
  --mappo-target-kl 0.05 \
  --mappo-mid-eval-every 30 \
  --obs-variant spawn_aware
```
