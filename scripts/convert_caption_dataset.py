"""Convert origin/target caption JSONs to qattack sample format.

Origin captions format:
  [
    {"image": "0.jpg", "caption": ["...", "...", ...]},
    ...
  ]

Target captions format:
  [
    {"image": "0.jpg", "caption": ["...", "...", ...]},
    ...
  ]

Outputs:
  - samples JSON for qattack dataset
  - optional direct_test_pairs.jsonl with clean=adv for direct test mode

Usage:
  python scripts/convert_caption_dataset.py \
    --origin_captions examples/origin_images/caption.json \
    --target_captions examples/target_images/caption.json \
    --clean-image-root examples/origin_images \
    --adv-image-root examples/target_images \
    --output examples/caption_samples.json \
    --pairs-out examples/caption_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert caption datasets to qattack samples")
    parser.add_argument("--origin_captions", default="examples/origin_images/qwen2_captions.json", help="Path to origin captions JSON")
    parser.add_argument("--target_captions", default="examples/target_images/qwen2_captions_target.json", help="Path to target captions JSON")
    parser.add_argument("--clean-image-root", default="examples/origin_images/", help="Clean image root directory")
    parser.add_argument("--adv-image-root", default="examples/target_images/", help="Adversarial image root directory")
    parser.add_argument("--output", default="examples/caption_samples.json", help="Output samples JSON")
    parser.add_argument("--pairs-out", default="examples/caption_pairs.jsonl", help="Output pairs JSONL (clean=adv)")
    parser.add_argument("--prompt", default="Describe the image in a simple sentence.")
    parser.add_argument("--target-policy", choices=["first", "concat", "none"], default="first")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only keep the first N samples (after optional skip-missing).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples when clean/adv image file is missing",
    )
    return parser.parse_args()


def build_target_text(captions: List[str], policy: str) -> Optional[str]:
    if not captions:
        return None
    if policy == "none":
        return None
    if policy == "concat":
        return " / ".join(captions)
    return captions[2]


def build_caption_index(data: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    caption_map: Dict[str, List[str]] = {}
    caption_stem_map: Dict[str, str] = {}

    for item in data:
        image_name = item.get("image")
        if not image_name:
            continue
        image_name = str(image_name)
        captions = [str(c) for c in (item.get("caption", []) or [])]
        caption_map[image_name] = captions
        caption_stem_map[Path(image_name).stem] = image_name

    return caption_map, caption_stem_map


def lookup_captions(
    image_name: str,
    caption_map: Dict[str, List[str]],
    caption_stem_map: Dict[str, str],
) -> Tuple[Optional[str], Optional[List[str]]]:
    image_name = str(image_name)

    if image_name in caption_map:
        return image_name, caption_map[image_name]

    alt_name = caption_stem_map.get(Path(image_name).stem)
    if alt_name and alt_name in caption_map:
        return alt_name, caption_map[alt_name]

    return None, None


def resolve_clean_image(clean_root: Path, image_name: str) -> Path:
    candidate = clean_root / image_name
    if candidate.exists():
        return candidate

    stem = Path(image_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        alt = clean_root / f"{stem}{ext}"
        if alt.exists():
            return alt

    return candidate


def numeric_sort_key(path: Path):
    import re

    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    m = re.search(r"\d+", stem)
    if m:
        return (0, int(m.group()), path.name)
    return (1, stem, path.name)


def main() -> None:
    args = parse_args()

    origin_captions_path = Path(args.origin_captions).resolve()
    target_captions_path = Path(args.target_captions).resolve()
    clean_image_root = Path(args.clean_image_root).resolve()
    adv_image_root = Path(args.adv_image_root).resolve()
    repo_root = Path.cwd().resolve()
    output_path = Path(args.output).resolve()

    with origin_captions_path.open("r", encoding="utf-8") as f:
        origin_data = json.load(f)
    with target_captions_path.open("r", encoding="utf-8") as f:
        target_data = json.load(f)

    if not isinstance(origin_data, list):
        raise ValueError("origin_captions JSON must be a list")
    if not isinstance(target_data, list):
        raise ValueError("target_captions JSON must be a list")

    origin_caption_map, origin_caption_stem_map = build_caption_index(origin_data)
    target_caption_map, target_caption_stem_map = build_caption_index(target_data)

    samples: List[Dict[str, Any]] = []

    missing_clean = 0
    missing_adv = 0
    missing_origin_caption = 0
    missing_target_caption = 0

    adv_files = sorted(
        [p for p in adv_image_root.iterdir() if p.is_file()],
        key=numeric_sort_key,
    )

    for adv_idx, adv_path in enumerate(adv_files):
        adv_image_name = adv_path.name

        # 先用 adv 图名在 origin captions 里找对应原图
        origin_image_name, origin_captions = lookup_captions(
            adv_image_name,
            origin_caption_map,
            origin_caption_stem_map,
        )
        if origin_captions is None or origin_image_name is None:
            missing_origin_caption += 1
            continue

        # target_text 来自 target_captions，优先按 adv 图名匹配，失败再按 origin 图名匹配
        target_image_name, target_captions = lookup_captions(
            adv_image_name,
            target_caption_map,
            target_caption_stem_map,
        )
        if target_captions is None:
            target_image_name, target_captions = lookup_captions(
                origin_image_name,
                target_caption_map,
                target_caption_stem_map,
            )
        if target_captions is None or target_image_name is None:
            missing_target_caption += 1
            continue

        clean_abs = resolve_clean_image(clean_image_root, origin_image_name)

        has_adv = adv_path.exists()
        if not has_adv:
            missing_adv += 1
            continue

        has_clean = clean_abs.exists()
        if not has_clean:
            missing_clean += 1
            if args.skip_missing:
                continue

        sample_id = f"caption_{adv_idx:04d}"
        try:
            clean_rel_root = clean_image_root.relative_to(repo_root)
        except ValueError:
            clean_rel_root = Path(clean_image_root.name)

        try:
            adv_rel_root = adv_image_root.relative_to(repo_root)
        except ValueError:
            adv_rel_root = Path(adv_image_root.name)

        clean_rel_image_path = str(clean_rel_root / clean_abs.name)
        adv_rel_image_path = str(adv_rel_root / adv_path.name)

        target_text = build_target_text(target_captions, args.target_policy)

        samples.append(
            {
                "sample_id": sample_id,
                "image_path": clean_rel_image_path,
                "adv_image_path": adv_rel_image_path,
                "user_prompt": args.prompt,
                "target_text": target_text,
                "category": "caption",
                "meta": {
                    "source": "caption_dataset",
                    "origin_image": origin_image_name,
                    "target_image": target_image_name,
                    "adv_image": adv_image_name,
                    "num_origin_captions": len(origin_captions),
                    "num_target_captions": len(target_captions),
                    "has_clean": has_clean,
                    "has_adv": has_adv,
                },
            }
        )

        if args.max_samples is not None and len(samples) >= args.max_samples:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    if args.pairs_out:
        pairs_path = Path(args.pairs_out).resolve()
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        with pairs_path.open("w", encoding="utf-8") as f:
            for s in samples:
                clean_path = str(Path(s["image_path"]))
                adv_path = str(Path(s["adv_image_path"]))

                if args.skip_missing:
                    if not (Path(args.clean_image_root) / Path(clean_path).name).exists():
                        continue
                if not (Path(args.adv_image_root) / Path(adv_path).name).exists():
                    continue

                f.write(
                    json.dumps(
                        {
                            "sample_id": s["sample_id"],
                            "clean_image_path": clean_path,
                            "adv_image_path": adv_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "num_samples": len(samples),
        "missing_origin_caption": missing_origin_caption,
        "missing_target_caption": missing_target_caption,
        "missing_clean": missing_clean,
        "missing_adv": missing_adv,
        "skip_missing": bool(args.skip_missing),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
