"""Factory methods for constructing attack instances from config."""

from __future__ import annotations

from typing import Any, Dict

from .base_attack import BaseAttack
from .visual_pgd import VisualPGDAttack


def create_attack(attack_config: Dict[str, Any]) -> BaseAttack:
    """Create attack instance from configuration.

    Supported examples:
      - {"name": "visual_pgd", "steps": 20, ...}
      - {"type": "pgd", "norm": "linf", ...}
    """
    attack_type = attack_config.get("name") or attack_config.get("type") or "visual_pgd"
    norm_type = str(attack_type).lower().replace("_", "-")

    if norm_type in {"visual-pgd", "visualpgd", "pgd"}:
        return VisualPGDAttack(attack_config)

    raise ValueError(f"Unsupported attack type: {attack_type}")
