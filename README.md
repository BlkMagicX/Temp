# QuantJB

## Boundary Drift Evaluation

This repository now includes a minimal boundary-drift evaluation pipeline for clean-harmful multimodal inputs.

Run:

python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_eval.yaml

Artifacts:

- outputs/<exp_name>/per_sample/boundary_drift_per_sample.csv
- outputs/<exp_name>/summary/boundary_drift_summary.json

MM-SafetyBench run (new config):

python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_mm_safetybench.yaml

Useful subset overrides:

python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_mm_safetybench.yaml --set data.mm_safetybench.scenarios=["01-Illegal_Activitiy","02-HateSpeech"] --set data.mm_safetybench.image_variants=["SD"] --set data.mm_safetybench.limit_total=40

Attack/eval separation:

- All gradient-based operations (including perturbation generation and gradient metrics) run on `models.attack_model` (fixed bf16 backend).
- Quantized models in `models.eval_precisions` are used only for `m_q`, `delta_q`, and `flip` computation.
- For non-differentiable quant backends (for example GPTQ), set `metrics.quant_gradient_mode` to:
	- `not_available` (default): write `gradient_status = "not available for non-differentiable quant backend"`
	- `fp_surrogate`: reuse bf16 attack gradients and write `gradient_status = "fp surrogate approximation"`

Approximation note for SR metric:

The current first-layer perturbation survival metric uses an observable proxy representation (processor output pixel_values) and a fake quantization map inferred from precision naming (for example w4a16 -> 16-bit activation proxy). This is an approximation until unified first-layer pre/post-quant hooks are exposed by all quant backends.
