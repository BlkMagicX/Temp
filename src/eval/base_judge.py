"""Base interface for pluggable response judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseJudge(ABC):
    """Unified judge interface.

    Judge output should be serializable and include at least:
      - `refusal_flag` (bool)
      - `attack_success` (bool)
      - `matched_keywords` (list)
    """

    @abstractmethod
    def judge(self, sample: Any, model_response: str) -> Dict[str, Any]:
        """Evaluate one response for one sample.

        Args:
            sample: Sample object or dictionary.
            model_response: Model-generated text response.

        Returns:
            A dictionary with judge decisions and optional metadata.
        """
