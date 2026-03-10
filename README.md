# Reviewer Main Scripts

This folder is a compact reviewer-facing snapshot of the main experiment entrypoints used for the IROS 2026 submission. Most of the underlying training pipeline, model structure, and evaluation logic are inherited from or closely inspired by prior EndoViT and Cholec80 codebases and benchmarks, rather than introduced here as a new architecture.

The purpose of this release is therefore not to present a novel backbone or training framework, but to document the scripts used to study transferability between laparoscopic and robotic surgical domains. The files here are copies of the original entry scripts and config snapshots. They are not packaged as a standalone runnable project, and their imports still assume the original repository layout.

## Quick Access

| Resource | Location | Notes |
| --- | --- | --- |
| Box archive | `https://uofi.box.com/s/jbbomp8847ir6frw17ftw1fegjxyl20e` | Public share link for the full `IROS_2026_output` mirror |
| Robotic pretraining checkpoint | `path-to/best_ckpt_7_loss_0.1397.pth` | Robotic-only pretraining weights |
| Classic SPR checkpoint | `path-to/endovit_SPR.pth` | Classic SPR weights used for the other experiments |
| Fine-tuning outputs | `path-to/IROS_2026_output/` | Root of the archived fine-tuning runs |

## Included Files

| Path | Role | Primary use |
| --- | --- | --- |
| `endovit/pretraining/mae/main_pretrain.py` | Robotic MAE pretraining entrypoint | `E4` |
| `endovit/finetuning/surgical_phase_recognition/model/TeCNO/train.py` | Modified SPR fine-tuning entrypoint for SITL / OmniRAS-PR | `E1`, `E2` |
| `endovit/finetuning/surgical_phase_recognition/model/TeCNO/test_only.py` | Test-only evaluator for the robotic surgical-phase benchmark | `E1`, `E2` |
| `endovit/finetuning/surgical_phase_recognition/model/TeCNO/test_only_cholec80.py` | Cholec80-side evaluator from the modified pipeline | `E1` |
| `endovit/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_sitl.yml` | Main SITL fine-tuning config snapshot | `E2` |
| `endovit/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_mae.yml` | Cholec80 MAE fine-tuning config snapshot from the modified tree | reference |
| `endovit_vanilla/EndoViT/finetuning/surgical_phase_recognition/model/TeCNO/train.py` | Vanilla EndoViT SPR fine-tuning entrypoint | `E3` |
| `endovit_vanilla/EndoViT/test_only_cholec80.py` | Standalone Cholec80 test-only evaluator | `E1`, `E3` |
| `endovit_vanilla/EndoViT/finetuning/surgical_phase_recognition/model/TeCNO/modules/mae/config/config_feature_extract_mae.yml` | Vanilla Cholec80 fine-tuning config snapshot | `E3` |
| `endovit_vanilla/run_cholec80_sitl_injection_sweep.sh` | Injection sweep launcher | `E3` |
| `endovit_vanilla/EndoViT/pretraining/pretrained_endovit_models/pretraining_config.yml` | Reference config for classic EndoViT SPR pretraining | `E4` |

## Checkpoints And Archive

Large checkpoints were not duplicated into this folder.

| Artifact | Path | Notes |
| --- | --- | --- |
| Full fine-tuning archive mirror | `./IROS_2026_output/` | Mirrored in Box at `https://uofi.box.com/s/jbbomp8847ir6frw17ftw1fegjxyl20e` |
| Robotic pretraining weights | `path-to/best_ckpt_7_loss_0.1397.pth` | About 1.3 GB |
| Classic SPR weights | `path-to/endovit_SPR.pth` | About 1.3 GB |
| Freeze-setting runs | `path-to/IROS_2026_output/enodivt/` | Organized by freeze regime |
| Injection runs | `path-to/IROS_2026_output/sitl_cholec80_injection/` | Organized by injection ratio |

## `./IROS_2026_output` Layout

| Path from `./IROS_2026_output` | What it contains | Notes |
| --- | --- | --- |
| `enodivt/output_test_freeze_-1/` | Timestamped SITL runs for one freeze setting | Contains repeated runs such as `16_32-25.02.26_FeatureExtraction_sitl_phase/` |
| `enodivt/output_test_freeze_0/` | Timestamped SITL runs for another freeze setting | Same internal layout as above |
| `enodivt/output_test_freeze_1/` | Timestamped SITL runs for the third freeze setting | Same internal layout as above |
| `sitl_cholec80_injection/inject_0_0/` | Injection runs at ratio `0.0` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_0_10/` | Injection runs at ratio `0.10` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_0_25/` | Injection runs at ratio `0.25` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_0_50/` | Injection runs at ratio `0.50` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_0_65/` | Injection runs at ratio `0.65` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_0_80/` | Injection runs at ratio `0.80` | Multiple timestamped repeats |
| `sitl_cholec80_injection/inject_1_0/` | Injection runs at ratio `1.0` | Multiple timestamped repeats |
| `<timestamp>_FeatureExtraction_sitl_phase/` | Older direct run folders stored at archive root | Same structure as a normal run folder |

## Contents Of A Run Folder

| Relative path inside one run folder | Meaning |
| --- | --- |
| `checkpoints/epoch=...-val_acc_phase=....ckpt` | Saved model checkpoint |
| `checkpoints/phase_timeline.png` | Phase-timeline visualization exported during evaluation |
| `tb/version_0/hparams.yaml` | Exact run configuration and parsed arguments |
| `tb/version_0/events.out.tfevents...` | TensorBoard event logs |
| `cholec80_pickle_export/` | Optional export directory; in the current archive these folders are usually present but empty |

## How To Browse The Box Mirror

1. Open the Box share link and enter `IROS_2026_output/`.
2. For SITL fine-tuning runs, go to `enodivt/` and choose one of the `output_test_freeze_*` folders.
3. For injection experiments, go to `sitl_cholec80_injection/` and choose the desired `inject_*` ratio folder.
4. Inside a timestamped run folder, use `checkpoints/*.ckpt` for weights and `tb/version_0/hparams.yaml` for the exact configuration.
