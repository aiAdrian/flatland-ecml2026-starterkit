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

## Run

```bash
cd experimental/flatland_solver

python main.py --mode eval --policy random --episodes 3
python main.py --mode eval --policy dla --episodes 3
python main.py --mode eval --policy dla --episodes 1 --rendering

# train BC and evaluate from checkpoint
python main.py --mode train --policy bc --episodes 2 --train-epochs 1
python main.py --mode eval --policy bc --episodes 2

# train MAPPO and evaluate from checkpoint
python main.py --mode train --policy mappo --episodes 2 --train-epochs 1
python main.py --mode eval --policy mappo --episodes 2
```
