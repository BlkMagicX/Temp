"""HarmBench judge placeholder adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

_base_module = importlib.import_module("src.eval.base_judge")
BaseJudge = getattr(_base_module, "BaseJudge")


class HarmBenchJudge(BaseJudge):
    """Placeholder interface for HarmBench classifier integration.

    Current status:
      - Defines external-model loading and inference entrypoints.
      - Raises clear exceptions when classifier path/runtime is unavailable.
      - Does not fabricate internal classifier behavior.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: str = "cuda",
        lazy_load: bool = True,
    ) -> None:
        self.classifier_path = classifier_path
        self.device = device
        self._classifier = None

        if not lazy_load:
            self._load_classifier_or_raise()

    def judge(self, sample: Any, model_response: str) -> Dict[str, Any]:
        """Run HarmBench judge inference.

        Raises:
            FileNotFoundError: If classifier weights are missing.
            NotImplementedError: If inference integration is not implemented yet.
        """
        self._load_classifier_or_raise()

        # TODO: Replace this with actual HarmBench classifier inference.
        # Expected return should include refusal_flag/attack_success and optional
        # classifier scores for analysis.
        raise NotImplementedError(
            "HarmBenchJudge inference is not implemented yet. " "Please integrate the official classifier tokenizer/model pipeline."
        )

    def _load_classifier_or_raise(self) -> None:
        """Load classifier weights or raise a clear error."""
        if self._classifier is not None:
            return

        if not self.classifier_path:
            raise FileNotFoundError("HarmBenchJudge requires `classifier_path` in config, but it is missing.")

        path = Path(self.classifier_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HarmBench classifier weights not found: {path}. " "Please provide a valid local path before using HarmBenchJudge."
            )

        # TODO: instantiate classifier model/tokenizer here.
        # Keep a sentinel to indicate loader path reached.
        self._classifier = "PLACEHOLDER_CLASSIFIER_HANDLE"
