"""Boundary drift evaluation experiment for quantized VLM safety analysis."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.algorithm.anchor_scorer import AnchorScorer, AnchorScoringConfig
from src.algorithm.boundary_metrics import boundary_drift, flip_indicator, resolve_boundary_tau, select_boundary_near_safe
from src.algorithm.refusal_margin import compute_margin
from src.algorithm.sensitivity import compute_kappa, compute_margin_gradient
from src.algorithm.survival import FirstLayerSurvivalConfig, build_small_delta, compute_survival_rate
from src.data.clean_harmful_dataset import CleanHarmfulDataset, CleanHarmfulSample
from src.data.mm_safetybench_dataset import MMSafetyBenchDataset
from src.eval.boundary_drift_evaluator import BoundaryDriftResultWriter
from src.models.model_factory import create_vlm

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):
        return iterable


class BoundaryDriftExperiment:
    """Run boundary-drift evaluation on clean-harmful multimodal samples."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.exp_name = str(config.get("exp_name", "boundary_drift_eval"))

        out_cfg = dict(config.get("output", {}))
        self.output_root = str(out_cfg.get("root_dir", "outputs"))
        self.writer = BoundaryDriftResultWriter(output_root=self.output_root, exp_name=self.exp_name)

        self.data_cfg = dict(config.get("data", {}))
        self.batch_size = int(self.data_cfg.get("batch_size", 1))
        self.dataset_type = str(self.data_cfg.get("dataset_type", "clean_harmful")).lower()
        if self.dataset_type in {"clean_harmful", "clean-harmful"}:
            self.dataset = CleanHarmfulDataset(
                annotation_path=self.data_cfg["annotation_path"],
                image_root=self.data_cfg.get("image_root"),
                check_image_exists=bool(self.data_cfg.get("check_image_exists", True)),
            )
        elif self.dataset_type in {"mm_safetybench", "mm-safetybench", "mm_safety_bench"}:
            mm_cfg = dict(self.data_cfg.get("mm_safetybench", {}))
            if not mm_cfg:
                raise ValueError("data.mm_safetybench config is required when dataset_type=mm_safetybench")
            self.dataset = MMSafetyBenchDataset.from_config(
                mm_cfg,
                check_image_exists=bool(self.data_cfg.get("check_image_exists", True)),
            )
        else:
            raise ValueError(f"Unsupported data.dataset_type: {self.dataset_type}")

        anchors_cfg = dict(config.get("anchors", {}))
        self.anchor_cfg = AnchorScoringConfig(
            refusal_anchors=list(anchors_cfg.get("refusal_anchors", [])),
            compliance_anchors=list(anchors_cfg.get("compliance_anchors", [])),
            aggregation=str(anchors_cfg.get("aggregation", "logsumexp")),
        )
        if not self.anchor_cfg.refusal_anchors or not self.anchor_cfg.compliance_anchors:
            raise ValueError("anchors.refusal_anchors and anchors.compliance_anchors must be non-empty")
        self.anchor_scorer = AnchorScorer(self.anchor_cfg)

        anchors_tau_cfg = dict(anchors_cfg.get("tau", {}))
        if not anchors_tau_cfg:
            anchors_tau_cfg = dict(config.get("tau", {}))
        self.anchor_tau_mode = str(anchors_tau_cfg.get("mode", "fixed"))
        self.anchor_tau_fixed = float(anchors_tau_cfg.get("fixed", 1.0))
        self.anchor_tau_quantile = float(anchors_tau_cfg.get("quantile", 0.2))

        tau_cfg = dict(config.get("tau", {}))
        self.tau_mode = str(tau_cfg.get("mode", "fixed"))
        self.tau_fixed = float(tau_cfg.get("fixed", 1.0))
        self.tau_quantile = float(tau_cfg.get("quantile", 0.2))

        metrics_cfg = dict(config.get("metrics", {}))
        self.enable_grad_metrics = bool(metrics_cfg.get("enable_grad_metrics", False))
        self.grad_on_boundary_only = bool(metrics_cfg.get("grad_on_boundary_only", True))
        self.enable_survival = bool(metrics_cfg.get("enable_survival", False))
        self.survival_on_boundary_only = bool(metrics_cfg.get("survival_on_boundary_only", True))
        self.survival_cfg = FirstLayerSurvivalConfig(
            delta_type=str(metrics_cfg.get("survival_delta_type", "random")),
            delta_budget=float(metrics_cfg.get("survival_delta_budget", 1.0 / 255.0)),
            eps=float(metrics_cfg.get("survival_eps", 1e-8)),
        )
        self.quant_gradient_mode = str(metrics_cfg.get("quant_gradient_mode", "not_available")).lower()
        if self.quant_gradient_mode not in {"not_available", "fp_surrogate"}:
            raise ValueError("metrics.quant_gradient_mode must be one of: not_available/fp_surrogate")

        models_cfg = dict(config.get("models", {}))
        self.eval_template = dict(models_cfg.get("eval_model_template", {}))
        if not self.eval_template:
            raise ValueError("models.eval_model_template is required")

        self.eval_precisions = list(models_cfg.get("eval_precisions", []))
        if not self.eval_precisions:
            raise ValueError("models.eval_precisions must contain at least one quantized precision")

        self.precision_overrides = dict(models_cfg.get("precision_overrides", {}))
        self.fp_precision_mode = str(models_cfg.get("fp_precision_mode", "bf16"))
        self.attack_model_template = dict(models_cfg.get("attack_model", {}))

        # Explicit separation required by experiment design.
        self.attack_model: Optional[Any] = None

    def _build_attack_model_cfg(self) -> Dict[str, Any]:
        """Build attack model config (fixed bf16 backend)."""
        cfg = dict(self.eval_template)
        cfg.update(self.attack_model_template)
        cfg["precision_mode"] = self.fp_precision_mode
        cfg["backend_type"] = "bf16"
        cfg.pop("quant_model_path", None)
        return cfg

    def _build_eval_model_cfg(self, precision: str) -> Dict[str, Any]:
        cfg = dict(self.eval_template)
        cfg["precision_mode"] = precision

        override = dict(self.precision_overrides.get(precision, {}))
        if "quant_backend_config" in override:
            merged_qcfg = dict(cfg.get("quant_backend_config", {}))
            merged_qcfg.update(override.get("quant_backend_config") or {})
            cfg["quant_backend_config"] = merged_qcfg
            override.pop("quant_backend_config", None)
        for k, v in override.items():
            cfg[k] = v
        if "backend_type" not in cfg:
            cfg["backend_type"] = "bf16" if precision == "bf16" else "gptq"

        return cfg

    @staticmethod
    def _is_non_differentiable_backend(backend_type: str) -> bool:
        return backend_type.lower() in {"gptq", "vllm"}

    def _prepare_attack_model(self) -> None:
        self.attack_model = self._load_attack_model()

    def _load_attack_model(self) -> Any:
        cfg = self._build_attack_model_cfg()
        model = create_vlm(cfg)
        model.load_model()
        return model

    def _load_single_eval_model(self, precision: str) -> tuple[Any, Dict[str, Any]]:
        cfg = self._build_eval_model_cfg(precision)
        model = create_vlm(cfg)
        model.load_model()
        return model, cfg

    @staticmethod
    def _release_model(model: Optional[Any]) -> None:
        if model is None:
            return

        if hasattr(model, "model"):
            model.model = None
        if hasattr(model, "processor"):
            model.processor = None
        if hasattr(model, "vllm_llm"):
            model.vllm_llm = None

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load_image_tensor(self, sample: CleanHarmfulSample, device: torch.device) -> torch.Tensor:
        image_path = self._resolve_sample_image_path(sample)
        img = Image.open(image_path).convert("RGB")
        return self._pil_to_unit_tensor(img).unsqueeze(0).to(device)

    def _resolve_sample_image_path(self, sample: CleanHarmfulSample) -> Path:
        if hasattr(self.dataset, "annotation_path"):
            dataset_parent = self.dataset.annotation_path.parent
            image_root = getattr(self.dataset, "image_root", None)
            return sample.resolve_image_path(dataset_parent, image_root)

        if hasattr(self.dataset, "dataset_root"):
            return sample.resolve_image_path(Path(self.dataset.dataset_root), None)

        return sample.resolve_image_path(Path.cwd(), None)

    def _compute_margin_for_model(self, model: Any, image_tensor: torch.Tensor, sample: CleanHarmfulSample, model_key: str) -> float:
        refusal_scores = self.anchor_scorer.score_refusal(
            model=model,
            image_tensor=image_tensor,
            prompt=sample.prompt,
            cache_prefix=sample.sample_id,
            model_key=model_key,
        )
        compliance_scores = self.anchor_scorer.score_compliance(
            model=model,
            image_tensor=image_tensor,
            prompt=sample.prompt,
            cache_prefix=sample.sample_id,
            model_key=model_key,
        )
        return compute_margin(
            refusal_scores=refusal_scores,
            compliance_scores=compliance_scores,
            aggregation=self.anchor_cfg.aggregation,
            tau_mode=self.anchor_tau_mode,
            tau_fixed=self.anchor_tau_fixed,
            tau_quantile=self.anchor_tau_quantile,
        )

    def _iter_batches(self, items: List[CleanHarmfulSample]) -> List[List[CleanHarmfulSample]]:
        return [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    def run(self) -> Dict[str, Any]:
        self._prepare_attack_model()

        attack_model = self.attack_model
        if attack_model is None:
            raise RuntimeError("Failed to load attack model")

        samples = list(self.dataset.samples)
        sample_fp_margin: Dict[str, float] = {}

        # Stage-1: compute m_fp(z) only with attack_model (bf16).
        for sample in tqdm(samples, desc="Stage-1 FP margins", dynamic_ncols=True):
            img = self._load_image_tensor(sample, attack_model.get_device())
            m_fp = self._compute_margin_for_model(attack_model, img, sample, model_key="attack_fp")
            sample_fp_margin[sample.sample_id] = m_fp

        tau = resolve_boundary_tau(
            m_fp_values=sample_fp_margin.values(),
            mode=("quantile" if self.tau_mode == "quantile" else "fixed"),
            fixed_tau=self.tau_fixed,
            quantile=self.tau_quantile,
        )
        boundary_flags = select_boundary_near_safe([sample_fp_margin[s.sample_id] for s in samples], tau=tau)
        sample_boundary_near = {s.sample_id: boundary_flags[i] for i, s in enumerate(samples)}

        rows: List[Dict[str, Any]] = []

        # FP rows for complete per-sample traceability.
        for sample in samples:
            m_fp = sample_fp_margin[sample.sample_id]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "model_precision": "fp",
                    "category": sample.category,
                    "source": sample.source,
                    "is_clean_harmful": sample.is_clean_harmful,
                    "m_fp": m_fp,
                    "m_q": m_fp,
                    "delta_q": 0.0,
                    "flip": 0,
                    "boundary_near": sample_boundary_near[sample.sample_id],
                    "kappa": None,
                    "eta": None,
                    "rho": None,
                    "survival_rate": None,
                    "gradient_status": None,
                    "error": None,
                }
            )

        attack_grad_cache: Dict[str, torch.Tensor] = {}
        attack_kappa_cache: Dict[str, float] = {}
        survival_cache: Dict[Tuple[str, str], float] = {}

        # Stage-1.5: precompute attack-side gradient and perturbation artifacts.
        need_any_grad = self.enable_grad_metrics or (self.enable_survival and self.survival_cfg.delta_type == "fgsm")
        if need_any_grad:
            for sample in tqdm(samples, desc="Stage-1.5 gradients", dynamic_ncols=True):
                sid = sample.sample_id
                use_sample = bool(sample_boundary_near[sid]) or (not self.grad_on_boundary_only)
                if not use_sample:
                    continue

                img = self._load_image_tensor(sample, attack_model.get_device())
                grad = compute_margin_gradient(
                    model=attack_model,
                    image_tensor=img,
                    prompt=sample.prompt,
                    refusal_anchors=self.anchor_cfg.refusal_anchors,
                    compliance_anchors=self.anchor_cfg.compliance_anchors,
                    aggregation=self.anchor_cfg.aggregation,
                    aggregation_tau=self.anchor_tau_fixed if self.anchor_tau_mode == "fixed" else 1.0,
                )
                attack_grad_cache[sid] = grad
                attack_kappa_cache[sid] = compute_kappa(grad)

        if self.enable_survival:
            for sample in tqdm(samples, desc="Stage-1.5 survival", dynamic_ncols=True):
                sid = sample.sample_id
                use_sample = bool(sample_boundary_near[sid]) or (not self.survival_on_boundary_only)
                if not use_sample:
                    continue

                img = self._load_image_tensor(sample, attack_model.get_device())
                delta = build_small_delta(
                    image_tensor=img,
                    cfg=self.survival_cfg,
                    margin_grad=attack_grad_cache.get(sid),
                )
                for precision in self.eval_precisions:
                    survival_cache[(sid, precision)] = compute_survival_rate(
                        model=attack_model,
                        precision_name=precision,
                        image_tensor=img,
                        prompt=sample.prompt,
                        delta=delta,
                        eps=self.survival_cfg.eps,
                    )

        self._release_model(attack_model)
        attack_model = None

        # Stage-2: compute m_q(z), delta_q(z), F_q(z) with eval models only.
        for precision in tqdm(self.eval_precisions, desc="Stage-2 precisions", dynamic_ncols=True):
            model, model_cfg = self._load_single_eval_model(precision)

            backend_type = str(model_cfg.get("backend_type", "")).lower()
            non_diff_backend = self._is_non_differentiable_backend(backend_type)

            for sample in tqdm(samples, desc=f"Stage-2 {precision}", dynamic_ncols=True, leave=False):
                sid = sample.sample_id
                m_fp = sample_fp_margin[sid]
                boundary_near = sample_boundary_near[sid]

                row: Dict[str, Any] = {
                    "sample_id": sid,
                    "model_precision": precision,
                    "category": sample.category,
                    "source": sample.source,
                    "is_clean_harmful": sample.is_clean_harmful,
                    "m_fp": m_fp,
                    "m_q": None,
                    "delta_q": None,
                    "flip": None,
                    "boundary_near": boundary_near,
                    "kappa": None,
                    "eta": None,
                    "rho": None,
                    "survival_rate": None,
                    "gradient_status": None,
                    "error": None,
                }

                img_q = self._load_image_tensor(sample, model.get_device())
                m_q = self._compute_margin_for_model(model, img_q, sample, model_key=precision)
                row["m_q"] = m_q
                row["delta_q"] = boundary_drift(m_q=m_q, m_fp=m_fp)
                row["flip"] = flip_indicator(m_fp=m_fp, m_q=m_q)

                need_grad = self.enable_grad_metrics and (boundary_near or (not self.grad_on_boundary_only))
                if need_grad:
                    if sid in attack_kappa_cache and (precision == "bf16" or not non_diff_backend):
                        row["kappa"] = attack_kappa_cache.get(sid)
                        row["eta"] = 0.0
                        row["rho"] = 1.0
                        row["gradient_status"] = "attack_model_fp_gradient"
                    elif sid in attack_kappa_cache and non_diff_backend:
                        if self.quant_gradient_mode == "fp_surrogate":
                            row["kappa"] = attack_kappa_cache.get(sid)
                            row["eta"] = 0.0
                            row["rho"] = 1.0
                            row["gradient_status"] = "fp surrogate approximation"
                        else:
                            row["gradient_status"] = "not available for non-differentiable quant backend"

                need_sr = self.enable_survival and (boundary_near or (not self.survival_on_boundary_only))
                if need_sr:
                    row["survival_rate"] = survival_cache.get((sid, precision))

                rows.append(row)

            self._release_model(model)

        per_sample_csv = self.writer.write_per_sample_csv(rows)
        summary = self.writer.build_summary(rows=rows, tau=tau)
        summary_path = self.writer.write_summary_json(summary)

        return {
            "exp_name": self.exp_name,
            "output_dir": str(self.writer.exp_dir),
            "per_sample_csv": str(per_sample_csv),
            "summary_json": str(summary_path),
        }


def load_experiment_config(config_path: str | Path) -> Dict[str, Any]:
    """Load JSON or YAML config file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Top-level config must be a dictionary")
        return cfg

    raise ValueError("Unsupported config format, use .json/.yaml/.yml")


def run_experiment(config_path: str | Path) -> Dict[str, Any]:
    config = load_experiment_config(config_path)
    return run_experiment_from_config(config)


def run_experiment_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runner = BoundaryDriftExperiment(config=config)
    return runner.run()
