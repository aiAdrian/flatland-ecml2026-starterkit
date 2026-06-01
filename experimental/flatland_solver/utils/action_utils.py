from typing import Dict


def normalize_actions(action_dict: Dict[int, object]) -> Dict[int, int]:
    normalized: Dict[int, int] = {}
    for handle, action in action_dict.items():
        if hasattr(action, "value"):
            normalized[handle] = int(action.value)
        else:
            normalized[handle] = int(action)
    return normalized
