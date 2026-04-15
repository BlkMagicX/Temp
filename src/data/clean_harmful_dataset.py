"""Dataset loader for manually curated clean-harmful multimodal samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CleanHarmfulSample:
    """Schema for dangerous-but-clean (non-explicit-jailbreak) inputs."""

    sample_id: str
    image_path: str
    prompt: str
    category: str
    source: str
    is_clean_harmful: bool
    notes: Optional[str] = None

    def resolve_image_path(self, dataset_parent: Path, image_root: Optional[Path] = None) -> Path:
        """Resolve absolute image path from sample record."""
        p = Path(self.image_path)
        if p.is_absolute():
            return p
        if image_root is not None:
            return (image_root / p).resolve()
        return (dataset_parent / p).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dict."""
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "category": self.category,
            "source": self.source,
            "is_clean_harmful": self.is_clean_harmful,
            "notes": self.notes,
        }


class CleanHarmfulDataset:
    """Read clean-harmful annotations from JSON/JSONL.

    Required fields:
        sample_id, image_path, prompt, category, source, is_clean_harmful
    Optional fields:
        notes
    """

    def __init__(
        self,
        annotation_path: Union[str, Path],
        image_root: Optional[Union[str, Path]] = None,
        check_image_exists: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path).resolve()
        self.image_root = Path(image_root).resolve() if image_root is not None else None
        self.check_image_exists = check_image_exists

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        suffix = self.annotation_path.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError("annotation_path must be .json or .jsonl")

        self.samples: List[CleanHarmfulSample] = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CleanHarmfulSample:
        return self.samples[idx]

    @staticmethod
    def _require_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        raise ValueError(f"Field {field_name} must be bool")

    def _load_samples(self) -> List[CleanHarmfulSample]:
        items = self._read_items(self.annotation_path)
        out: List[CleanHarmfulSample] = []
        seen = set()

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Record at index={i} must be an object")

            sample = CleanHarmfulSample(
                sample_id=str(item.get("sample_id", "")).strip(),
                image_path=str(item.get("image_path", "")).strip(),
                prompt=str(item.get("prompt", "")).strip(),
                category=str(item.get("category", "")).strip(),
                source=str(item.get("source", "")).strip(),
                is_clean_harmful=self._require_bool(item.get("is_clean_harmful", False), "is_clean_harmful"),
                notes=(str(item.get("notes")).strip() if item.get("notes") is not None else None),
            )

            if not sample.sample_id:
                raise ValueError(f"Missing sample_id at index={i}")
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)

            if not sample.image_path:
                raise ValueError(f"Missing image_path for sample_id={sample.sample_id}")
            if not sample.prompt:
                raise ValueError(f"Missing prompt for sample_id={sample.sample_id}")
            if not sample.category:
                raise ValueError(f"Missing category for sample_id={sample.sample_id}")
            if not sample.source:
                raise ValueError(f"Missing source for sample_id={sample.sample_id}")

            if self.check_image_exists:
                image_abs = sample.resolve_image_path(self.annotation_path.parent, self.image_root)
                if not image_abs.exists():
                    raise FileNotFoundError(f"Image not found for sample_id={sample.sample_id}: {image_abs}")

            out.append(sample)

        return out

    @staticmethod
    def _read_items(path: Path) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON annotations must be a list")
            return data

        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return out
