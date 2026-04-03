"""Dataset reader for JSON/JSONL multimodal samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .sample_schema import VLMSample


class VLMSafetyDataset:
    """Simple dataset for VLM safety samples.

    Features:
        - Read `.json` (list of objects) and `.jsonl` (one object per line)
        - Required/optional field validation through `VLMSample`
        - Optional image path existence checks
        - `__len__` and `__getitem__` for experiment pipelines

    This class intentionally avoids coupling to specific benchmarks and can be
    adapted for HarmBench / MM-SafetyBench format mapping.
    """

    def __init__(
        self,
        annotation_path: Union[str, Path],
        image_root: Optional[Union[str, Path]] = None,
        check_image_exists: bool = True,
        return_dict: bool = False,
    ) -> None:
        """Initialize dataset and load samples.

        Args:
            annotation_path: Path to `.json` or `.jsonl` sample annotations.
            image_root: Optional root directory for relative image paths.
            check_image_exists: Whether to verify image files during load.
            return_dict: If True, `__getitem__` returns dict; otherwise returns
                `VLMSample` object.

        Raises:
            FileNotFoundError: If annotation file is missing.
            ValueError: If annotation format is unsupported or invalid.
        """
        self.annotation_path = Path(annotation_path).resolve()
        self.image_root = Path(image_root).resolve() if image_root is not None else None
        self.check_image_exists = check_image_exists
        self.return_dict = return_dict

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        if self.annotation_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("annotation_path must be a .json or .jsonl file")

        self.samples: List[VLMSample] = self._load_samples()

    def __len__(self) -> int:
        """Return number of loaded samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[VLMSample, Dict[str, Any]]:
        """Return one sample by index.

        Args:
            idx: Sample index.

        Returns:
            `VLMSample` or dictionary based on `return_dict`.
        """
        sample = self.samples[idx]
        return sample.to_dict() if self.return_dict else sample

    def _load_samples(self) -> List[VLMSample]:
        """Load and validate all records from annotation file."""
        raw_items = self._read_annotation_items(self.annotation_path)
        samples: List[VLMSample] = []

        seen_ids = set()
        for i, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Each record must be an object, got type={type(item)} at index={i}")

            sample = VLMSample.from_dict(item)

            if sample.sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id detected: {sample.sample_id}")
            seen_ids.add(sample.sample_id)

            resolved_path = sample.resolve_image_path(
                image_root=self.image_root,
                annotation_parent=self.annotation_path.parent,
            )
            if self.check_image_exists and not resolved_path.exists():
                raise FileNotFoundError(f"Image file not found for sample_id={sample.sample_id}: {resolved_path}")

            samples.append(sample)

        return samples

    @staticmethod
    def _read_annotation_items(path: Path) -> List[Dict[str, Any]]:
        """Read raw records from .json/.jsonl and return as list of dicts."""
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON annotation must be a list of sample objects")
            return data

        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                items.append(obj)
        return items
