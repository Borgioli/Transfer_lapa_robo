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

Checkpoints can be found: https://uofi.box.com/s/jbbomp8847ir6frw17ftw1fegjxyl20e


- Robotic pretraining checkpoint used for the robotic-only setup:
  - `path-to/best_ckpt_7_loss_0.1397.pth`
- Classic SPR checkpoint used for the other experiments:
  - `path-to/endovit_SPR.pth`
- Fine-tuned experiment outputs and checkpoints:
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_-1`
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_0`
  - `path-to/IROS_2026_output/enodivt/output_test_freeze_1`
  - `path-to/IROS_2026_output/sitl_cholec80_injection`

### `./IROS_2026_output` structure

- `./IROS_2026_output/enodivt/`
  - Main SITL fine-tuning runs grouped by freeze setting.
  - The three subfolders are:
    - `output_test_freeze_-1/`
    - `output_test_freeze_0/`
    - `output_test_freeze_1/`
  - Each of those contains multiple timestamped run folders, for example:
    - `./IROS_2026_output/enodivt/output_test_freeze_-1/16_32-25.02.26_FeatureExtraction_sitl_phase/`
  - Inside one run folder, the useful files are usually:
    - `checkpoints/epoch=...-val_acc_phase=....ckpt`
    - `checkpoints/phase_timeline.png`
    - `tb/version_0/hparams.yaml`
    - `tb/version_0/events.out.tfevents...`
  - Many runs also contain `cholec80_pickle_export/`; in the current archive these directories appear to be present as placeholders and are typically empty.

- `./IROS_2026_output/sitl_cholec80_injection/`
  - E3 injection experiments grouped by laparoscopic injection ratio.
  - The ratio folders are:
    - `inject_0_0/`
    - `inject_0_10/`
    - `inject_0_25/`
    - `inject_0_50/`
    - `inject_0_65/`
    - `inject_0_80/`
    - `inject_1_0/`
  - Each ratio folder contains several timestamped repeat runs, for example:
    - `./IROS_2026_output/sitl_cholec80_injection/inject_0_25/06_02-23.02.26_FeatureExtraction_sitl_inject_0_25/`
  - Inside each run folder, the useful files are usually:
    - `checkpoints/epoch=...-val_acc_phase=....ckpt`
    - `checkpoints/phase_timeline.png`
    - `tb/version_0/hparams.yaml`
    - `tb/version_0/events.out.tfevents...`
  - `hparams.yaml` is the easiest place to inspect the exact arguments used for a particular run.

- `./IROS_2026_output/<timestamp>_FeatureExtraction_sitl_phase/`
  - In addition to the organized subtrees above, the archive root also contains a number of older direct run folders named like `08_39-17.02.26_FeatureExtraction_sitl_phase`.
  - These have the same internal layout as a normal run folder:
    - `checkpoints/`
    - `tb/version_0/`
    - sometimes `cholec80_pickle_export/`

### How to navigate the Box mirror

- Start at `IROS_2026_output/`.
- For frozen or unfrozen SITL fine-tuning results, open `enodivt/` and then one of `output_test_freeze_-1/`, `output_test_freeze_0/`, or `output_test_freeze_1/`.
- For injection ablations, open `sitl_cholec80_injection/`, then the desired `inject_*` ratio folder, then a timestamped run folder.
- Within a run folder:
  - the model weights are in `checkpoints/*.ckpt`
  - the run config is in `tb/version_0/hparams.yaml`
  - TensorBoard logs are in `tb/version_0/events.out.tfevents...`

Both pretrained `.pth` files above are about 1.3 GB each, so they should be linked externally if you want them accessible in the shared reviewer repo.
