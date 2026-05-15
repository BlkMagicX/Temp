if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
echo "Cannot find conda.sh"
exit 1
fi

conda activate gptq
echo "[1/4] quant env: boundary_drift_mm_safetybench_w"
python src/main.py --config configs/boundary_drift_mm_safetybench_w.yaml

echo "[2/4] quant env: representation_drift_all_layers"
python scripts/run_representation_drift.py --config configs/representation_drift_all_layers.yaml

echo "[3/4] quant env: qcsd_landscape_mm_safetybench_w3a16_layer4_top20"
python scripts/run_qcsd_landscape.py --config configs/qcsd_landscape_mm_safetybench_w3a16_layer4_top20.yaml

echo "[4/4] quant env: qcsd_landscape_mm_safetybench_w3a16_layer31_top20"
python scripts/run_qcsd_landscape.py --config configs/qcsd_landscape_mm_safetybench_w3a16_layer31_top20.yaml