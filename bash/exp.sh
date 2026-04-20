if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
echo "Cannot find conda.sh"
exit 1
fi

# conda activate gptq
# echo "[1/3] quant env: boundary_drift_mm_safetybench_w"
# python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_mm_safetybench_w.yaml

conda activate quant
echo "[2/3] quant env: boundary_drift_mm_safetybench_w4a4"
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_mm_safetybench_w4a4.yaml

echo "[3/3] quant env: boundary_drift_mm_safetybench_w8a8"
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_boundary_drift_eval.py --config configs/boundary_drift_mm_safetybench_w8a8.yaml