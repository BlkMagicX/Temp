"""Experiment 1: generate adversarial perturbations on bf16 and evaluate transfer across precisions."""

from __future__ import annotations

import json
import importlib
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

_dataset_module = importlib.import_module("src.data.dataset")
VLMSafetyDataset = getattr(_dataset_module, "VLMSafetyDataset")

_writer_module = importlib.import_module("src.eval.result_writer")
ResultWriter = getattr(_writer_module, "ResultWriter")

_judge_factory_module = importlib.import_module("src.eval.judge_factory")
create_judge = getattr(_judge_factory_module, "create_judge")

_factory_module = importlib.import_module("src.models.model_factory")
create_vlm = getattr(_factory_module, "create_vlm")


@dataclass
class AttackArtifacts:
    """Artifacts generated during adversarial sample creation."""

    adv_image: Image.Image
    adv_image_path: Path
    delta_linf: float
    delta_l2: float
    attack_runtime: float


class TransferAcrossPrecisionExperiment:
    """Runner for Experiment-1 transfer evaluation.

        Pipeline:
            1) Load dataset
            2) Load bf16 attack model (white-box)
            3) Generate adversarial image per sample
            4) Evaluate same adversarial image on bf16/w8a8/w4a16
            5) Save per-sample JSONL

    Notes:
      - Current `attack_success` and `refusal_flag` are rule-based placeholders.
      - Quantized model loading depends on backend integration in model wrapper.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.exp_name = str(config.get("exp_name", "exp_transfer_across_precision"))

        output_root = config.get("output", {}).get("root_dir", "outputs")
        self.writer = ResultWriter(base_output_dir=output_root, exp_name=self.exp_name)

        data_cfg = config.get("data", {})
        self.dataset = VLMSafetyDataset(
            annotation_path=data_cfg["annotation_path"],
            image_root=data_cfg.get("image_root"),
            check_image_exists=bool(data_cfg.get("check_image_exists", True)),
            return_dict=False,
        )

        models_cfg = config.get("models", {})
        self.eval_precisions = models_cfg.get("eval_precisions", ["bf16", "w8a8", "w4a16", "w8a16", "w4a4"])
        self.eval_model_template = dict(models_cfg.get("eval_model_template", {}))

        self.generation_cfg = config.get("generation", {"max_new_tokens": 128, "do_sample": False})
        self.judge_cfg = dict(config.get("judge", {"type": "rule_based"}))
        self.runtime_cfg = dict(config.get("runtime", {}))
        self.resume = bool(self.runtime_cfg.get("resume", True))
        self.direct_image_test_cfg = dict(self.runtime_cfg.get("direct_image_test", {}))
        self.enable_direct_image_test = bool(self.direct_image_test_cfg.get("enabled", False))
        self.direct_test_image_path = self.direct_image_test_cfg.get("image_path")
        self.direct_test_pairs = self._load_direct_test_pairs(self.direct_image_test_cfg)
        self.direct_test_pairs_by_sample_id = {str(p["sample_id"]): p for p in self.direct_test_pairs if p.get("sample_id")}

        self.eval_models: Dict[str, Optional[Any]] = {}
        self.eval_model_errors: Dict[str, str] = {}
        self.judge = create_judge(self.judge_cfg)

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment and return run metadata, with progress bar."""
        from tqdm import tqdm

        if not self.enable_direct_image_test:
            raise ValueError(
                "Direct-image test mode is required. " "Please set runtime.direct_image_test.enabled=true and provide pairs_file or pairs."
            )
        if not self.direct_test_image_path and not self.direct_test_pairs:
            raise ValueError(
                "runtime.direct_image_test.enabled=true requires either runtime.direct_image_test.image_path "
                "or runtime.direct_image_test.pairs / runtime.direct_image_test.pairs_file"
            )

        self._prepare_eval_models()
        existing_keys = self.writer.load_existing_keys(["model_precision", "sample_id"]) if self.resume else set()

        dataset_list = list(self.dataset)
        pbar = tqdm(dataset_list, desc=f"Evaluating {self.exp_name}", ncols=100)
        for sample in pbar:
            missing_precisions = [p for p in self.eval_precisions if (p, sample.sample_id) not in existing_keys]
            if not missing_precisions:
                continue

            clean_image = self._load_clean_image(sample)
            artifacts = self._load_or_generate_adv(sample=sample)

            for precision in missing_precisions:
                pbar.set_postfix({"sample_id": sample.sample_id, "precision": precision})
                record = self._evaluate_one_precision(
                    precision=precision,
                    sample=sample,
                    clean_image=clean_image,
                    adv_image=artifacts.adv_image,
                    adv_image_path=artifacts.adv_image_path,
                    delta_linf=artifacts.delta_linf,
                    delta_l2=artifacts.delta_l2,
                    attack_runtime=artifacts.attack_runtime,
                )
                self.writer.append_per_sample_record(record)

        return {
            "exp_name": self.exp_name,
            "output_root": str(self.writer.exp_root),
            "per_sample_path": str(self.writer.per_sample_jsonl_path),
        }

    def _prepare_eval_models(self) -> None:
        """Load evaluation models for each precision and keep error states."""
        models_cfg = self.config.get("models", {})
        precision_overrides = models_cfg.get("precision_overrides", {})

        for precision in self.eval_precisions:
            cfg = dict(self.eval_model_template)
            cfg.setdefault("name", "qwen2vl")
            cfg["precision_mode"] = precision

            # 应用每个 precision 的专属覆盖配置
            override_cfg = precision_overrides.get(precision, {})
            if override_cfg:
                if "quant_backend_config" in override_cfg:
                    base_qcfg = dict(cfg.get("quant_backend_config", {}))
                    base_qcfg.update(override_cfg["quant_backend_config"] or {})
                    cfg["quant_backend_config"] = base_qcfg

                for k, v in override_cfg.items():
                    if k == "quant_backend_config":
                        continue
                    cfg[k] = v

            # fallback
            if "backend_type" not in cfg or cfg["backend_type"] is None:
                cfg["backend_type"] = "bf16" if precision == "bf16" else "awq"

            try:
                model = create_vlm(cfg)
                model.load_model()
                self.eval_models[precision] = model
            except Exception as exc:  # noqa: BLE001
                self.eval_models[precision] = None
                self.eval_model_errors[precision] = str(exc)

    def _resolve_sample_image_path(self, sample: Any) -> Path:
        """Resolve sample image path with schema-aware priority."""
        if self.enable_direct_image_test:
            pair = self._get_direct_test_pair_for_sample(sample)
            if pair and pair.get("clean_image_path"):
                return self._resolve_existing_image_path(self._resolve_direct_test_path(str(pair["clean_image_path"])))

        if hasattr(sample, "resolve_image_path"):
            return self._resolve_existing_image_path(
                sample.resolve_image_path(
                    image_root=self.dataset.image_root,
                    annotation_parent=self.dataset.annotation_path.parent,
                )
            )

        p = Path(sample["image_path"] if isinstance(sample, dict) else sample.image_path)
        if p.is_absolute():
            return self._resolve_existing_image_path(p)

        if getattr(self.dataset, "image_root", None) is not None:
            return self._resolve_existing_image_path((self.dataset.image_root / p).resolve())

        return self._resolve_existing_image_path((self.dataset.annotation_path.parent / p).resolve())

    @staticmethod
    def _resolve_existing_image_path(path: Path) -> Path:
        """Try to resolve missing image path by swapping common extensions."""
        if path.exists():
            return path

        suffix = path.suffix.lower()
        candidates = []
        if suffix in {".jpg", ".jpeg", ".png"}:
            for ext in (".jpg", ".jpeg", ".png"):
                if ext != suffix:
                    candidates.append(path.with_suffix(ext))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return path

    def _load_clean_image(self, sample: Any) -> Image.Image:
        """Load clean image as RGB PIL image.

        Image path resolution matches `VLMSample` behavior when possible.
        """
        p = self._resolve_sample_image_path(sample)
        return Image.open(p).convert("RGB")

    def _load_or_generate_adv(
        self,
        sample: Any,
    ) -> AttackArtifacts:
        """Load adversarial image directly from provided paths (no persistence)."""
        sample_id = sample.sample_id

        clean_image = self._load_clean_image(sample)

        return self._load_or_prepare_direct_test_adv(
            sample_id=sample_id,
            clean_image=clean_image,
        )

    def _load_or_prepare_direct_test_adv(
        self,
        sample_id: str,
        clean_image: Image.Image,
    ) -> AttackArtifacts:
        """Use a user-provided image path as adv image (skip attack generation)."""
        source_path = Path(str(self.direct_test_image_path)).resolve()
        pair = self._get_direct_test_pair_for_sample({"sample_id": sample_id})
        if pair and pair.get("adv_image_path"):
            source_path = self._resolve_direct_test_path(str(pair["adv_image_path"]))

        if not source_path.exists():
            raise FileNotFoundError(f"direct_image_test image not found: {source_path}")

        adv_image = Image.open(source_path).convert("RGB")
        delta_linf, delta_l2 = self._compute_delta_metrics(clean_image=clean_image, adv_image=adv_image)
        attack_runtime = 0.0

        return AttackArtifacts(
            adv_image=adv_image,
            adv_image_path=source_path,
            delta_linf=delta_linf,
            delta_l2=delta_l2,
            attack_runtime=attack_runtime,
        )

    def _load_direct_test_pairs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load direct-test pairs from inline config or external json/jsonl file.

        Pair schema:
            {
              "sample_id": "demo_0001",  # recommended
              "clean_image_path": "...",
              "adv_image_path": "..."
            }
        """
        pairs: List[Dict[str, Any]] = []

        inline_pairs = cfg.get("pairs", []) or []
        if not isinstance(inline_pairs, list):
            raise ValueError("runtime.direct_image_test.pairs must be a list")
        for idx, item in enumerate(inline_pairs):
            pairs.append(self._normalize_direct_test_pair(item, source=f"pairs[{idx}]"))

        pairs_file = cfg.get("pairs_file")
        if pairs_file:
            file_path = self._resolve_direct_test_path(str(pairs_file))
            if not file_path.exists():
                raise FileNotFoundError(f"direct_image_test.pairs_file not found: {file_path}")

            suffix = file_path.suffix.lower()
            if suffix == ".json":
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("direct_image_test.pairs_file(.json) must contain a list")
                for idx, item in enumerate(data):
                    pairs.append(self._normalize_direct_test_pair(item, source=f"{file_path.name}[{idx}]"))
            elif suffix == ".jsonl":
                with file_path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        pairs.append(self._normalize_direct_test_pair(item, source=f"{file_path.name}:L{line_no}"))
            else:
                raise ValueError("direct_image_test.pairs_file only supports .json or .jsonl")

        return pairs

    def _normalize_direct_test_pair(self, item: Any, source: str) -> Dict[str, Any]:
        """Validate and normalize one direct-test pair record."""
        if not isinstance(item, dict):
            raise ValueError(f"Invalid direct test pair at {source}: expected object")

        clean_path = item.get("clean_image_path")
        adv_path = item.get("adv_image_path")
        if not clean_path or not adv_path:
            raise ValueError(f"Invalid direct test pair at {source}: both clean_image_path and adv_image_path are required")

        sample_id = item.get("sample_id")
        clean_path_str = str(clean_path)
        adv_path_str = str(adv_path)

        # 兼容旧 pairs：adv 路径写成 "mpcattack/..." 时自动补上 "examples/"
        if not Path(adv_path_str).is_absolute() and adv_path_str.replace("\\", "/").startswith("mpcattack/"):
            adv_path_str = str(Path("examples") / Path(adv_path_str))

        normalized = {
            "sample_id": str(sample_id) if sample_id is not None else None,
            "clean_image_path": clean_path_str,
            "adv_image_path": adv_path_str,
        }
        return normalized

    def _get_direct_test_pair_for_sample(self, sample: Any) -> Optional[Dict[str, Any]]:
        """Get direct-test pair for sample, matched by sample_id when available."""
        if not self.direct_test_pairs:
            return None

        sample_id = None
        if isinstance(sample, dict):
            sample_id = sample.get("sample_id")
        else:
            sample_id = getattr(sample, "sample_id", None)

        if sample_id is not None:
            pair = self.direct_test_pairs_by_sample_id.get(str(sample_id))
            if pair is None:
                raise ValueError(
                    f"No direct_image_test pair found for sample_id={sample_id}. "
                    "Please provide matching sample_id entries in direct_image_test.pairs/pairs_file."
                )
            return pair

        raise ValueError("Sample has no sample_id; cannot match direct_image_test pairs")

    def _resolve_direct_test_path(self, path_value: str) -> Path:
        """Resolve direct-test image path from absolute or repo-relative path."""
        p = Path(path_value)
        if p.is_absolute():
            return p
        return (Path.cwd() / p).resolve()

    def _compute_delta_metrics(self, clean_image: Image.Image, adv_image: Image.Image) -> tuple[float, float]:
        """Compute delta metrics; resize adv image for metric compatibility if needed."""
        if adv_image.size != clean_image.size:
            adv_metric_img = adv_image.resize(clean_image.size, Image.BICUBIC)
        else:
            adv_metric_img = adv_image

        clean = self._pil_to_unit_tensor(clean_image)
        adv = self._pil_to_unit_tensor(adv_metric_img)
        delta = (adv - clean).float()
        delta_linf = float(delta.abs().max().item())
        delta_l2 = float(delta.view(-1).norm(p=2).item())
        return delta_linf, delta_l2

    def _evaluate_one_precision(
        self,
        precision: str,
        sample: Any,
        clean_image: Image.Image,
        adv_image: Image.Image,
        adv_image_path: Path,
        delta_linf: float,
        delta_l2: float,
        attack_runtime: float,
    ) -> Dict[str, Any]:
        """Evaluate clean/adv behavior under one precision model."""
        t0 = time.perf_counter()
        model = self.eval_models.get(precision)

        clean_response = ""
        adv_response = ""
        clean_loss: Optional[float] = None
        adv_loss: Optional[float] = None
        model_error = self.eval_model_errors.get(precision)

        if model is not None:
            try:
                clean_response = model.generate_response(clean_image, sample.user_prompt, self.generation_cfg)
                adv_response = model.generate_response(adv_image, sample.user_prompt, self.generation_cfg)
                # clean_loss = self._compute_loss_from_image(model, clean_image, sample.user_prompt, sample.target_text)
                # adv_loss = self._compute_loss_from_image(model, adv_image, sample.user_prompt, sample.target_text)
            except Exception as exc:  # noqa: BLE001
                model_error = str(exc)

        judge_sample: Any = sample
        if isinstance(sample, dict):
            judge_sample = dict(sample)
            judge_sample["clean_response"] = clean_response
            judge_sample["adv_response"] = adv_response
        else:
            judge_sample = {
                "sample_id": getattr(sample, "sample_id", None),
                "target_text": getattr(sample, "target_text", None),
                "clean_response": clean_response,
                "adv_response": adv_response,
            }

        judge_result = self.judge.judge(sample=judge_sample, clean_response=clean_response, adv_response=adv_response)

        runtime = float(time.perf_counter() - t0)

        return {
            "model_precision": precision,
            "sample_id": sample.sample_id,
            "clean_image_path": str(self._resolve_sample_image_path(sample)),
            "clean_response": clean_response,
            "adv_response": adv_response,
            "delta_linf": delta_linf,
            "delta_l2": delta_l2,
            "runtime": runtime,
            "attack_runtime": attack_runtime,
            "adv_image_path": str(adv_image_path),
            "judge": judge_result,
            "error": model_error,
        }

    @staticmethod
    def _compute_loss_from_image(
        model: Any,
        image: Image.Image,
        prompt: str,
        target_text: Optional[str],
    ) -> Optional[float]:
        """Compute scalar loss from image via model differentiable forward API."""
        pixel = TransferAcrossPrecisionExperiment._pil_to_unit_tensor(image).unsqueeze(0).to(model.get_device())
        outputs = model.forward_for_loss(pixel, prompt=prompt, target_text=target_text)
        loss = outputs.get("loss")
        if loss is None:
            return None
        return float(loss.detach().item())

    @staticmethod
    def _pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
        """Convert RGB PIL image to float tensor in [0, 1], shape (C, H, W)."""
        import numpy as np

        img = image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    @staticmethod
    def _image_md5(image: Image.Image) -> str:
        """Compute md5 of RGB image bytes for traceability."""
        img = image.convert("RGB")
        return hashlib.md5(img.tobytes()).hexdigest()


def load_experiment_config(config_path: str | Path) -> Dict[str, Any]:
    """Load experiment config from JSON or YAML.

    YAML loading is optional and requires `pyyaml`.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("YAML config requires pyyaml. Please install pyyaml.") from exc

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Top-level YAML config must be a dictionary")
        return cfg

    raise ValueError("Unsupported config format. Use .json, .yaml, or .yml")


def run_experiment(config_path: str | Path) -> Dict[str, Any]:
    """Convenience entrypoint for running this experiment."""
    config = load_experiment_config(config_path)
    runner = TransferAcrossPrecisionExperiment(config=config)
    return runner.run()
