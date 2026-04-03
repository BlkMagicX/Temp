"""Unified sample schema and validation utilities for multimodal safety experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VLMSample:
    """Canonical sample record used across attack and evaluation pipelines.

    Required fields:
        - `sample_id`
        - `image_path`
        - `user_prompt`

    Optional fields:
        - `target_text`
        - `category`
        - `meta`
    """

    sample_id: str
    image_path: str
    user_prompt: str
    target_text: Optional[str] = None
    category: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate_basic(self) -> None:
        """Validate basic field type and non-empty constraints.

        Raises:
            ValueError: If required fields are missing or invalid.
            TypeError: If field types are invalid.
        """
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("`sample_id` must be a non-empty string")

        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise ValueError("`image_path` must be a non-empty string")

        if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
            raise ValueError("`user_prompt` must be a non-empty string")

        if self.target_text is not None and not isinstance(self.target_text, str):
            raise TypeError("`target_text` must be a string or null")

        if self.category is not None and not isinstance(self.category, str):
            raise TypeError("`category` must be a string or null")

        if not isinstance(self.meta, dict):
            raise TypeError("`meta` must be a dictionary")

    def resolve_image_path(
        self,
        image_root: Optional[Path] = None,
        annotation_parent: Optional[Path] = None,
    ) -> Path:
        """Resolve image path to an absolute path.

        Resolution priority:
        1) absolute `image_path`
        2) `image_root / image_path` when `image_root` is provided
        3) `annotation_parent / image_path` when `annotation_parent` is provided
        4) current working directory / `image_path`

        Returns:
            Absolute resolved image path.
        """
        img = Path(self.image_path)

        if img.is_absolute():
            return img

        if image_root is not None:
            return (image_root / img).resolve()

        if annotation_parent is not None:
            return (annotation_parent / img).resolve()

        return img.resolve()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMSample":
        """Create `VLMSample` from dictionary and validate basic fields."""
        sample = cls(
            sample_id=str(data.get("sample_id", "")).strip(),
            image_path=str(data.get("image_path", "")).strip(),
            user_prompt=str(data.get("user_prompt", "")).strip(),
            target_text=data.get("target_text"),
            category=data.get("category"),
            meta=data.get("meta") if data.get("meta") is not None else {},
        )
        sample.validate_basic()
        return sample

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to plain dictionary."""
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "user_prompt": self.user_prompt,
            "target_text": self.target_text,
            "category": self.category,
            "meta": self.meta,
        }
