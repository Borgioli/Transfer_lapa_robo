#!/usr/bin/env bash
set -e

# Reviewer note:
# This launcher encodes the structure of the E3 injection sweep. Replace the
# placeholder paths below before running it in a local clone.

# ── Change to endovit_vanilla root (required for relative sys.path in models/mae.py) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/EndoViT"
echo "Working directory: $(pwd)"

# ── Paths ──────────────────────────────────────────────────────────────────
# `DATA_ROOT` is the laparoscopic project root used by the vanilla pipeline,
# while `SITL_ROOT` points to the robotic dataset injected during E3.
CONFIG="./finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_mae.yml"
TRAIN_SCRIPT="./finetuning/surgical_phase_recognition/model/TeCNO/train.py"
DATA_ROOT="path-to/project-root"
OUTPUT_BASE="./finetuning/surgical_phase_recognition/output_dir/cholec80_sitl_injection"
SITL_ROOT="path-to/SITL_phases"

# ── Common training arguments ──────────────────────────────────────────────
MAE_MODEL="vit_base_patch16"
MAE_CKPT="path-to/endovit_SPR.pth"
SEED=1665
SAVE_TOP_K=1
LEARNING_RATE=0.0005
MAE_LAYER_DECAY=0.65
MAE_WEIGHT_DECAY=0.
MAE_REINIT_N_LAYERS=-1
FREEZE_WEIGHTS=-1
MODEL_SPECIFIC_BATCH_SIZE_MAX=128

# ── Injection ratios to sweep ─────────────────────────────────────────────
# Each ratio creates a separate `inject_*` output subtree under OUTPUT_BASE.
#RATIOS=(0.0 0.10 0.25 0.50 1.0)
#RATIOS=(0.0 0.10 0.25 0.50 0.65 0.80 1.0)
RATIOS=(0.80)

echo "============================================================"
echo " SITL → Cholec80 Injection Sweep"
echo " Config : ${CONFIG}"
echo " Ratios : ${RATIOS[*]}"
echo " Output : ${OUTPUT_BASE}"
echo "============================================================"
echo ""

for RATIO in "${RATIOS[@]}"; do

    # Friendly tag for the output directory
    RATIO_TAG=$(echo "${RATIO}" | sed 's/\./_/g')
    OUTPUT_PATH="${OUTPUT_BASE}/inject_${RATIO_TAG}"

    echo "------------------------------------------------------------"
    echo " Starting experiment: sitl_inject_ratio = ${RATIO}"
    echo " Output: ${OUTPUT_PATH}"
    echo "------------------------------------------------------------"

    COMMON_ARGS=(
        -c "${CONFIG}"
        --data_root "${DATA_ROOT}"
        --mae_model "${MAE_MODEL}"
        --mae_ckpt "${MAE_CKPT}"
        --learning_rate "${LEARNING_RATE}"
        --freeze_weights "${FREEZE_WEIGHTS}"
        --mae_layer_decay "${MAE_LAYER_DECAY}"
        --mae_weight_decay "${MAE_WEIGHT_DECAY}"
        --mae_reinit_n_layers "${MAE_REINIT_N_LAYERS}"
        --return_mae_optimizer_groups
        --model_specific_batch_size_max "${MODEL_SPECIFIC_BATCH_SIZE_MAX}"
        --seed "${SEED}"
        --output_path "${OUTPUT_PATH}"
        --save_top_k "${SAVE_TOP_K}"
        --loss_balancing
        --wandb_project_name "EndoViT_Cholec80_SITL_Injection"
        --wandb_tags "FeatureExtraction" "Cholec80" "SITL_Injection" "ratio_${RATIO_TAG}"
        --wandb_name_suffix "Cholec80_SITL_inject_${RATIO_TAG}"
    )

    if [ "${RATIO}" = "0.0" ]; then
        # Pure Cholec80 baseline — no injection flags needed
        python "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
    else
        python "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" \
            --sitl_inject \
            --sitl_inject_ratio "${RATIO}" \
            --sitl_data_root "${SITL_ROOT}"
    fi

    echo ""
    echo " ✓ Finished: sitl_inject_ratio = ${RATIO}"
    echo "============================================================"
    echo ""

done

echo "All ${#RATIOS[@]} injection experiments completed."
