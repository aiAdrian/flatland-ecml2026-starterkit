from pathlib import Path
import sys


def _patch_flatland_transition_api() -> None:
    from flatland.core.transition_map import GridTransitionMap

    if getattr(GridTransitionMap, "_legacy_signature_compat", False):
        return

    original = GridTransitionMap.get_transitions

    def compat(self, *args):
        if len(args) == 1:
            return original(self, args[0])
        if len(args) == 2:
            return original(self, (args[0], int(args[1])))
        if len(args) == 3:
            return original(self, ((int(args[0]), int(args[1])), int(args[2])))
        return original(self, *args)

    GridTransitionMap.get_transitions = compat
    GridTransitionMap._legacy_signature_compat = True


def _ensure_rl_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    rl_dir = repo_root / "reinforcement-learning"
    rl_path = str(rl_dir)
    if rl_dir.exists() and rl_path not in sys.path:
        sys.path.insert(0, rl_path)


def _ensure_legacy_obs_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    legacy_root = repo_root / "experimental" / "flatland_minimal_project_NOT_WORKING_TRANSFORM"
    legacy_dir = (
        legacy_root
        / "example"
        / "flatland_rail_env"
    )
    legacy_root_path = str(legacy_root)
    legacy_path = str(legacy_dir)
    if legacy_root.exists() and legacy_root_path not in sys.path:
        sys.path.insert(0, legacy_root_path)
    if legacy_dir.exists() and legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)


def BCObservationBuilder(obs_variant: str = "fast_tree"):
    obs_variant = str(obs_variant).lower()

    if obs_variant == "fast_tree":
        _ensure_rl_path()
        from my_observation_builder import FastTreeObsBuilder

        return FastTreeObsBuilder(max_depth=3, with_action_mask=True)

    _ensure_legacy_obs_path()
    _patch_flatland_transition_api()

    def _compat(cls):
        class CompatObs(cls):
            def _rail_get_transitions(self, pos, direction):
                p = self._pos_tuple(pos)
                return self.env.rail.get_transitions((p, int(direction)))

        CompatObs.__name__ = f"Compat{cls.__name__}"
        return CompatObs

    if obs_variant == "decision_point":
        from marl_attention_temporal_observation.decision_point_observation import DecisionPointObservation

        return _compat(DecisionPointObservation)(debug=False, search_depth=4)
    if obs_variant == "spawn_aware":
        from marl_attention_temporal_observation.spawn_aware_observation import SpawnAwareObservation

        return _compat(SpawnAwareObservation)(debug=False, search_depth=4)
    if obs_variant == "conflict_aware":
        from marl_attention_temporal_observation.conflict_aware_observation import ConflictAwareObservation

        return _compat(ConflictAwareObservation)(debug=False, search_depth=4, verbose_first_call=False)

    raise ValueError(f"Unsupported --obs-variant for BC: {obs_variant}")
