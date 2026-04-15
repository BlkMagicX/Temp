"""MM-SafetyBench adapter that outputs CleanHarmfulSample records."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .clean_harmful_dataset import CleanHarmfulSample


class MMSafetyBenchDataset:
    """Load MM-SafetyBench and map entries into CleanHarmfulSample format.

    Expected MM-SafetyBench layout:
      <dataset_root>/data/processed_questions/{scenario}.json
      <dataset_root>/data/imgs/{scenario}/{variant}/{question_id}.jpg

    A sample corresponds to one (scenario, question_id, image_variant) tuple.
    """

    DEFAULT_IMAGE_VARIANTS = ("SD", "SD_TYPO", "TYPO")

    def __init__(
        self,
        dataset_root: Union[str, Path],
        scenarios: Optional[Sequence[str]] = None,
        image_variants: Optional[Sequence[str]] = None,
        question_ids: Optional[Sequence[Union[str, int]]] = None,
        use_tiny_ids: bool = False,
        tiny_ids_path: Optional[Union[str, Path]] = None,
        limit_per_scenario: Optional[int] = None,
        limit_total: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
        check_image_exists: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.questions_root = self.dataset_root / "data" / "processed_questions"
        self.images_root = self.dataset_root / "data" / "imgs"
        self.check_image_exists = check_image_exists

        if not self.questions_root.exists():
            raise FileNotFoundError(f"processed_questions directory not found: {self.questions_root}")
        if not self.images_root.exists():
            raise FileNotFoundError(f"imgs directory not found: {self.images_root}")

        self._scenario_filter = {str(x) for x in (scenarios or [])}
        self._image_variants = [str(v) for v in (image_variants or self.DEFAULT_IMAGE_VARIANTS)]
        self._question_id_filter = {str(x) for x in (question_ids or [])}
        self._tiny_ids_by_scenario = self._load_tiny_ids(use_tiny_ids=use_tiny_ids, tiny_ids_path=tiny_ids_path)
        self.limit_per_scenario = limit_per_scenario
        self.limit_total = limit_total
        self.shuffle = shuffle
        self.seed = seed

        self.samples: List[CleanHarmfulSample] = self._load_samples()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], check_image_exists: bool = True) -> "MMSafetyBenchDataset":
        """Build dataset from `data.mm_safetybench` config section."""
        return cls(
            dataset_root=cfg.get("dataset_root"),
            scenarios=cfg.get("scenarios"),
            image_variants=cfg.get("image_variants"),
            question_ids=cfg.get("question_ids"),
            use_tiny_ids=bool(cfg.get("use_tiny_ids", False)),
            tiny_ids_path=cfg.get("tiny_ids_path"),
            limit_per_scenario=cfg.get("limit_per_scenario"),
            limit_total=cfg.get("limit_total"),
            shuffle=bool(cfg.get("shuffle", False)),
            seed=int(cfg.get("seed", 42)),
            check_image_exists=check_image_exists,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CleanHarmfulSample:
        return self.samples[idx]

    def _iter_scenario_files(self) -> Iterable[Tuple[str, Path]]:
        files = sorted(self.questions_root.glob("*.json"))
        for path in files:
            scenario = path.stem
            if self._scenario_filter and scenario not in self._scenario_filter:
                continue
            yield scenario, path

    def _load_tiny_ids(
        self,
        use_tiny_ids: bool,
        tiny_ids_path: Optional[Union[str, Path]],
    ) -> Dict[str, Set[str]]:
        if not use_tiny_ids:
            return {}

        tiny_path = Path(tiny_ids_path).resolve() if tiny_ids_path else (self.dataset_root / "TinyVersion_ID_List.json")
        if not tiny_path.exists():
            raise FileNotFoundError(f"Tiny version id list not found: {tiny_path}")

        with tiny_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Supported schemas:
        # 1) {"01-Scenario": ["0", "1", ...], ...}
        # 2) ["0", "1", ...] (global ids for all scenarios)
        # 3) [{"Scenario": "01-...", "Sampled_ID_List": [5, 7, ...]}, ...]
        if isinstance(data, dict):
            out: Dict[str, Set[str]] = {}
            for scenario, ids in data.items():
                if isinstance(ids, list):
                    out[str(scenario)] = {str(x) for x in ids}
            return out

        if isinstance(data, list):
            # MM-SafetyBench default tiny file format: list of records.
            if data and all(isinstance(x, dict) for x in data):
                out: Dict[str, Set[str]] = {}
                for item in data:
                    scenario = str(item.get("Scenario", "")).strip()
                    ids = item.get("Sampled_ID_List")
                    if not scenario or not isinstance(ids, list):
                        continue
                    out[scenario] = {str(x) for x in ids}
                if out:
                    return out

            global_ids = {str(x) for x in data}
            return {"*": global_ids}

        raise ValueError("Unsupported TinyVersion_ID_List.json format")

    def _is_allowed_question_id(self, scenario: str, qid: str) -> bool:
        if self._question_id_filter and qid not in self._question_id_filter:
            return False

        if not self._tiny_ids_by_scenario:
            return True

        if scenario in self._tiny_ids_by_scenario:
            return qid in self._tiny_ids_by_scenario[scenario]

        if "*" in self._tiny_ids_by_scenario:
            return qid in self._tiny_ids_by_scenario["*"]

        return True

    @staticmethod
    def _resolve_prompt(item: Dict[str, Any], variant: str) -> str:
        if variant == "SD":
            prompt = str(item.get("Rephrased Question(SD)", "")).strip()
            if prompt:
                return prompt
        return str(item.get("Rephrased Question", "")).strip()

    def _load_samples(self) -> List[CleanHarmfulSample]:
        out: List[CleanHarmfulSample] = []

        for scenario, question_file in self._iter_scenario_files():
            with question_file.open("r", encoding="utf-8") as f:
                question_map = json.load(f)
            if not isinstance(question_map, dict):
                raise ValueError(f"Question file must be a dict: {question_file}")

            added_for_scenario = 0
            for qid in sorted(question_map.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
                qid_str = str(qid)
                if not self._is_allowed_question_id(scenario, qid_str):
                    continue

                item = question_map[qid]
                if not isinstance(item, dict):
                    continue

                for variant in self._image_variants:
                    prompt = self._resolve_prompt(item, variant)
                    if not prompt:
                        continue

                    image_abs = self.images_root / scenario / variant / f"{qid_str}.jpg"
                    if self.check_image_exists and not image_abs.exists():
                        continue

                    out.append(
                        CleanHarmfulSample(
                            sample_id=f"{scenario}:{variant}:{qid_str}",
                            image_path=str(image_abs),
                            prompt=prompt,
                            category=scenario,
                            source=f"MM-SafetyBench/{variant}",
                            is_clean_harmful=True,
                            notes=(str(item.get("Question", "")).strip() or None),
                        )
                    )
                    added_for_scenario += 1

                    if self.limit_per_scenario is not None and added_for_scenario >= int(self.limit_per_scenario):
                        break

                if self.limit_per_scenario is not None and added_for_scenario >= int(self.limit_per_scenario):
                    break

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(out)

        if self.limit_total is not None:
            out = out[: int(self.limit_total)]

        if not out:
            raise ValueError("No samples loaded from MM-SafetyBench with current filters")

        return out
