# Reviewer Main Scripts

This folder contains a curated copy of the main experiment entrypoints used while preparing the IROS 2026 submission. It is meant to help reviewers inspect the actual training and evaluation launch scripts without having to browse the full local `endovit/` and `endovit_vanilla/` trees.

These are entrypoints and config snapshots, not a standalone runnable package. Their internal imports still refer to the original project structure.

## Included scripts

- `endovit/pretraining/mae/main_pretrain.py`
  - Robotic MAE pretraining entrypoint used for the robotic-only pretraining setup.
- `endovit/finetuning/surgical_phase_recognition/model/TeCNO/train.py`
  - Modified SPR fine-tuning entrypoint used for the SITL / OmniRAS-PR experiments.
- `endovit/finetuning/surgical_phase_recognition/model/TeCNO/test_only.py`
  - Test-only evaluator for the robotic surgical-phase benchmark.
- `endovit/finetuning/surgical_phase_recognition/model/TeCNO/test_only_cholec80.py`
  - Cholec80-side evaluator from the modified pipeline.
- `endovit/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_sitl.yml`
  - Main SITL fine-tuning config snapshot.
- `endovit/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_mae.yml`
  - Cholec80 MAE fine-tuning config snapshot used in the modified tree.
- `endovit_vanilla/EndoViT/finetuning/surgical_phase_recognition/model/TeCNO/train.py`
  - Vanilla EndoViT SPR fine-tuning entrypoint used with the classic SPR checkpoint.
- `endovit_vanilla/EndoViT/test_only_cholec80.py`
  - Standalone Cholec80 test-only evaluator used for reviewer-facing evaluation.
- `endovit_vanilla/EndoViT/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_mae.yml`
  - Vanilla Cholec80 fine-tuning config snapshot.
- `endovit_vanilla/run_cholec80_sitl_injection_sweep.sh`
  - Sweep launcher for the Cholec80 plus SITL injection experiments.
- `endovit_vanilla/EndoViT/pretraining/pretrained_endovit_models/pretraining_config.yml`
  - Reference config for the classic EndoViT SPR pretraining setup.

## Experiment mapping

- `E1` cross-domain transfer:
  - `endovit/finetuning/.../train.py`
  - `endovit/finetuning/.../test_only.py`
  - `endovit_vanilla/EndoViT/test_only_cholec80.py`
- `E2` frozen vs partially unfrozen vs full fine-tuning:
  - `endovit/finetuning/.../train.py`
  - `endovit/finetuning/.../config_feature_extract_sitl.yml`
- `E3` laparoscopic data injection:
  - `endovit_vanilla/run_cholec80_sitl_injection_sweep.sh`
  - `endovit_vanilla/EndoViT/finetuning/.../train.py`
  - `endovit_vanilla/EndoViT/test_only_cholec80.py`
- `E4` robotic-only vs classic SPR pretraining:
  - `endovit/pretraining/mae/main_pretrain.py`
  - `endovit_vanilla/EndoViT/pretraining/pretrained_endovit_models/pretraining_config.yml`

## Checkpoint inventory

Large checkpoints were not duplicated into this folder.

- Robotic pretraining checkpoint used for the robotic-only setup:
  - `path-to/best_ckpt_7_loss_0.1397.pth`
- Classic SPR checkpoint used for the other experiments:
  - `path-to/endovit_SPR.pth`
- Fine-tuned experiment outputs and checkpoints:
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_-1`
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_0`
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_1`
  - `path-to/IROS_2026_output/sitl_cholec80_injection`

