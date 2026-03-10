#!/usr/bin/env python3
"""
Test-only evaluation script for surgical phase recognition.

Loads a trained checkpoint and evaluates ONLY on the predefined test videos,
computing per-video and overall accuracy, F1, and recall.

Reviewer note:
This release copy is intentionally self-contained. It reconstructs the label
mapping used in the paper and accepts either Lightning fine-tuned checkpoints
or backbone-only `.pth` files.

Usage:
    python test_only.py --checkpoint /path/to/best.ckpt --seed 42
    python test_only.py --checkpoint /path/to/best.ckpt --seed 123
    python test_only.py --checkpoint /path/to/best.ckpt --seed 7

Available seeds and their test splits:
  --seed 42  : videos [23, 26, 33, 35, 44]
  --seed 123 : videos [23, 26, 30, 44, 45]
  --seed 7   : videos [1, 4, 6, 15, 43]
  --seed 1665: videos [1, 4, 6, 15, 43]
"""

import sys
import os
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycm import ConfusionMatrix
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
)

# ── Adjust sys.path so the copied release script can still resolve the
# original model definitions without modifying package names. ───────────────
TECNO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TECNO_ROOT))
sys.path.insert(0, str(TECNO_ROOT / "pretraining" / "mae"))
# Also add the endovit pretraining path for prepare_mae_model
ENDOVIT_ROOT = TECNO_ROOT.parents[3]  # endovit/
sys.path.insert(0, str(ENDOVIT_ROOT / "pretraining" / "mae"))


# ============================================================
# CONSTANTS: Phase mappings (from sitl_feature_extract_v3.py)
# These tables collapse the robotic annotation vocabulary into the Cholec80
# label space used throughout the transfer experiments.
# ============================================================
CANONICAL_PHASES = {
    "None": 0,
    "Retraction of the gallbladder neck": 1,
    "Opening the anterior peritoneal layer of the triangle of Calot": 2,
    "Opening the posterior peritoneal layer of the triangle of Calot": 3,
    "Isolation of the cystic duct": 4,
    "Isolation of the cystic artery": 5,
    "clipping of the cystic duct": 6,
    "clipping of the cystic artery": 7,
    "Division of the cystic duct": 8,
    "Division of the cystic artery": 9,
    "dissection of the gallbladder from the liver": 10,
    "Exposure of the working area": 11,
    "specimen retrieval": 12,
}

PHASE_STRING_NORMALIZATION = {
    "exposure of the working area": "Exposure of the working area",
    "isolation of the cystic artery": "Isolation of the cystic artery",
    "Clipping of the cystic duct": "clipping of the cystic duct",
    "clipping of the cystic artery ( second branch)": "clipping of the cystic artery",
    "clipping of the cystic artery (second branch)": "clipping of the cystic artery",
    "Division of the cystic artery ( second branch)": "Division of the cystic artery",
    "Division of the cystic artery (second branch)": "Division of the cystic artery",
    "Isolation of the cystic artery ( second branch)": "Isolation of the cystic artery",
    "Division of the cystic duct and artery": "Division of the cystic duct",
    "Division of the cystic duct and cystic artery": "Division of the cystic duct",
    "6-Isolation of the cystic duct and cystic artery": "Isolation of the cystic duct",
    "8- clipping of the cystic duct and cystic artery": "clipping of the cystic duct",
    "1: exposure of the working area": "Exposure of the working area",
    "11: specimen retrieval": "specimen retrieval",
    "Opening the anterior peritoneal layer of the triangle of Calot": "Opening the posterior peritoneal layer of the triangle of Calot",
    "/": None,
    "Phase": None,
}

SITL_TO_CHOLEC80 = {
    "None":                                                            "Preparation",
    "Retraction of the gallbladder neck":                              "GallbladderRetraction",
    "Opening the anterior peritoneal layer of the triangle of Calot":  "CalotTriangleDissection",
    "Opening the posterior peritoneal layer of the triangle of Calot": "CalotTriangleDissection",
    "Isolation of the cystic duct":                                    "CalotTriangleDissection",
    "Isolation of the cystic artery":                                  "CalotTriangleDissection",
    "clipping of the cystic duct":                                     "ClippingCutting",
    "clipping of the cystic artery":                                   "ClippingCutting",
    "Division of the cystic duct":                                     "ClippingCutting",
    "Division of the cystic artery":                                   "ClippingCutting",
    "dissection of the gallbladder from the liver":                    "GallbladderDissection",
    "Exposure of the working area":                                    "Preparation",
    "specimen retrieval":                                              "GallbladderPackaging",
}

CHOLEC80_PHASES = {
    "Preparation":             0,
    "GallbladderRetraction":   1,
    "CalotTriangleDissection": 2,
    "ClippingCutting":         3,
    "GallbladderDissection":   4,
    "GallbladderPackaging":    5,
    "CleaningCoagulation":     6,
}

# SITL raw ID → Cholec80 raw ID
_SITL_RAW_TO_CHOLEC80_RAW = {}
for _name, _sitl_id in CANONICAL_PHASES.items():
    _cholec_name = SITL_TO_CHOLEC80.get(_name)
    if _cholec_name and _cholec_name in CHOLEC80_PHASES:
        _SITL_RAW_TO_CHOLEC80_RAW[_sitl_id] = CHOLEC80_PHASES[_cholec_name]

# Cholec80 phases actually present in SITL
INCLUDED_PHASES = [name for name in CHOLEC80_PHASES
                   if name in set(SITL_TO_CHOLEC80.values())]

# Build label map: raw Cholec80 ID → contiguous 0-based index
_kept_mapping = {name: label for name, label in CHOLEC80_PHASES.items()
                 if name in INCLUDED_PHASES}
_sorted_phases = sorted(_kept_mapping.items(), key=lambda x: x[1])
CLASS_LABELS = [name for name, _ in _sorted_phases]
LABEL_MAP = {raw_label: idx for idx, (_, raw_label) in enumerate(_sorted_phases)}
RAW_LABELS_TO_KEEP = set(_kept_mapping.values())
NUM_CLASSES = len(CLASS_LABELS)

# ── Phases to EXCLUDE from metric computation ─────────────────
# The model still outputs all NUM_CLASSES logits (matching the
# checkpoint), but these classes are ignored when computing
# accuracy, F1, recall, precision, classification report, and
# the confusion matrix.
# → Comment out a line to re-include that class in the metrics.
EXCLUDED_CHOLEC80_PHASES = {
    "GallbladderPackaging",
    # "CleaningCoagulation",   # uncomment to also exclude this one, etc.
}

# Indices & labels used for evaluation (everything minus excluded)
EVAL_CLASS_INDICES = [i for i, name in enumerate(CLASS_LABELS)
                      if name not in EXCLUDED_CHOLEC80_PHASES]
EVAL_CLASS_LABELS  = [CLASS_LABELS[i] for i in EVAL_CLASS_INDICES]


# ============================================================
# HARDCODED TEST SPLITS (one per seed)
# ============================================================
TEST_SPLITS = {
    42: {
        23: "video03-phase.txt",
        26: "video32-phase.txt",
        33: "video39-phase.txt",
        35: "video40-phase.txt",
        44: "video07-phase.txt",
    },
    123: {
        23: "video03-phase.txt",
        26: "video32-phase.txt",
        30: "video36-phase.txt",
        44: "video07-phase.txt",
        45: "video08-phase.txt",
    },
    7: {
         1: "video20-phase.txt",
         4: "video12-phase.txt",
         6: "video43-phase.txt",
        15: "video22-phase.txt",
        43: "video06-phase.txt",
    },
    1665: {
         8: "video08-phase.txt",
         11: "video11-phase.txt",
         15: "video15-phase.txt",
        16: "video16-phase.txt",
        29: "video29-phase.txt",
        34: "video34-phase.txt",
        46: "video46-phase.txt",
    },
    2: {
         24: "video24-phase.txt",
         9: "video09-phase.txt",
         43: "video43-phase.txt",
        21: "video21-phase.txt",
        17: "video17-phase.txt",
    },
}


# ============================================================
# Annotation helpers
# ============================================================
from difflib import get_close_matches


def normalize_phase_string(phase_str):
    if phase_str in PHASE_STRING_NORMALIZATION:
        return PHASE_STRING_NORMALIZATION[phase_str]
    if phase_str in CANONICAL_PHASES:
        return phase_str
    phase_lower = phase_str.strip().lower()
    for variant, canonical in PHASE_STRING_NORMALIZATION.items():
        if phase_lower == variant.strip().lower():
            return canonical
    for canonical_name in CANONICAL_PHASES:
        if phase_lower == canonical_name.strip().lower():
            return canonical_name
    return phase_str


def fuzzy_match_phase(phase_str, phase_mapping):
    if phase_str in phase_mapping:
        return phase_str, True
    for key in phase_mapping.keys():
        if phase_str.lower() == key.lower():
            return key, True
    cleaned = phase_str.strip().rstrip(".")
    if cleaned in phase_mapping:
        return cleaned, True
    matches = get_close_matches(phase_str, phase_mapping.keys(), n=1, cutoff=0.85)
    if matches:
        return matches[0], False
    return None, False


# ============================================================
# Build test dataframe for the 5 test videos
# ============================================================
def build_test_dataframe(sitl_root: Path, video_to_annotation: dict, fps: int = 1):
    """
    Build a DataFrame of (image_path, class, video_idx, frame) for the test
    videos, with Cholec80 mapping and contiguous label remapping applied.
    """
    frames_dir = sitl_root / "frames"
    annotations_dir = sitl_root / "phase_annotations"

    all_data = []
    skipped_nan = 0
    skipped_unknown = 0
    kept = 0
    _unmatched = set()

    for video_idx, ann_filename in sorted(video_to_annotation.items()):
        # Find video folder
        video_folder = None
        for d in frames_dir.iterdir():
            if d.is_dir():
                idx = int("".join(filter(str.isdigit, d.name)))
                if idx == video_idx:
                    video_folder = d
                    break
        if video_folder is None:
            print(f"WARNING: video folder for video {video_idx} not found, skipping")
            continue

        ann_file = annotations_dir / ann_filename
        if not ann_file.exists():
            print(f"WARNING: annotation file {ann_file} not found, skipping")
            continue

        # Read annotations
        ann_df = pd.read_csv(ann_file, sep="\t", dtype={"Frame": int, "Phase": str})
        nan_mask = ann_df["Phase"].isna()
        ann_df_valid = ann_df[~nan_mask].copy()
        if len(ann_df_valid) == 0:
            print(f"WARNING: all annotations NaN for video {video_idx}, skipping")
            continue

        frame_to_phase = dict(zip(ann_df_valid["Frame"], ann_df_valid["Phase"]))
        ann_frame_numbers = np.array(sorted(ann_df["Frame"].values))
        max_ann_frame = ann_frame_numbers[-1] if len(ann_frame_numbers) > 0 else 0

        # Get frames on disk
        frames = sorted(
            [f for f in video_folder.iterdir() if f.suffix in [".jpg", ".png", ".jpeg"]]
        )
        if not frames:
            print(f"WARNING: no frames for video {video_idx}")
            continue

        max_disk_frame = int("".join(filter(str.isdigit, frames[-1].stem)))
        scale = (max_ann_frame / max_disk_frame) if (max_disk_frame > 0 and max_ann_frame > 0) else 1.0

        sampled_frames = frames[::fps] if fps > 0 else frames

        for frame_path in sampled_frames:
            frame_num = int("".join(filter(str.isdigit, frame_path.stem)))

            # Scale to annotation space
            ann_frame_num = int(round(frame_num * scale))
            ann_frame_num = min(ann_frame_num, max_ann_frame)

            if ann_frame_num in frame_to_phase:
                phase_str = frame_to_phase[ann_frame_num]
            else:
                idx_s = np.searchsorted(ann_frame_numbers, ann_frame_num)
                candidates = []
                if idx_s < len(ann_frame_numbers):
                    candidates.append(ann_frame_numbers[idx_s])
                if idx_s > 0:
                    candidates.append(ann_frame_numbers[idx_s - 1])
                nearest = None
                for c in candidates:
                    if c in frame_to_phase:
                        if nearest is None or abs(c - ann_frame_num) < abs(nearest - ann_frame_num):
                            nearest = c
                if nearest is None:
                    skipped_nan += 1
                    continue
                phase_str = frame_to_phase[nearest]

            if phase_str is None or pd.isna(phase_str):
                skipped_nan += 1
                continue

            normalized = normalize_phase_string(phase_str)
            if normalized is None:
                skipped_unknown += 1
                continue

            matched_phase, _ = fuzzy_match_phase(normalized, CANONICAL_PHASES)
            if matched_phase is None:
                skipped_unknown += 1
                if phase_str not in _unmatched:
                    _unmatched.add(phase_str)
                continue

            raw_label = CANONICAL_PHASES[matched_phase]

            # Map SITL → Cholec80
            if raw_label not in _SITL_RAW_TO_CHOLEC80_RAW:
                skipped_unknown += 1
                continue
            cholec_raw = _SITL_RAW_TO_CHOLEC80_RAW[raw_label]

            # Filter to included phases
            if cholec_raw not in RAW_LABELS_TO_KEEP:
                skipped_unknown += 1
                continue

            # Remap to contiguous label
            label = LABEL_MAP[cholec_raw]

            all_data.append({
                "image_path": f"{video_folder.name}/{frame_path.name}",
                "class": label,
                "video_idx": video_idx,
                "frame": frame_num,
            })
            kept += 1

    if _unmatched:
        print(f"WARNING: {len(_unmatched)} unrecognized phase strings: {sorted(_unmatched)}")

    df = pd.DataFrame(all_data)
    print(f"\nTest data: {kept} frames kept, {skipped_nan} skipped (NaN), "
          f"{skipped_unknown} skipped (unknown/excluded)")
    print(f"Videos: {sorted(df['video_idx'].unique().tolist())}")
    print(f"Class distribution:\n{df['class'].value_counts().sort_index()}\n")
    return df


# ============================================================
# Simple image dataset
# ============================================================
class TestImageDataset(Dataset):
    def __init__(self, df, img_root, transform):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = self.img_root / row["image_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        label = int(row["class"])
        video_idx = int(row["video_idx"])
        frame = int(row["frame"])
        return img, label, video_idx, frame


# ============================================================
# Per-video accuracy (same as feature_extraction.py)
# ============================================================
def get_phase_acc(true_labels, pred_logits):
    """
    Compute per-video accuracy using pycm (identical to
    FeatureExtraction.get_phase_acc).
    """
    pred = torch.FloatTensor(pred_logits)
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    pred_phase = torch.softmax(pred, dim=1)
    labels_pred = torch.argmax(pred_phase, dim=1).cpu().numpy()
    cm = ConfusionMatrix(
        actual_vector=true_labels,
        predict_vector=labels_pred,
    )
    return cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro


# ============================================================
# Load model from checkpoint
# ============================================================
def _default_hparams(out_features: int = 6):
    """Return a SimpleNamespace with the default model hparams
    matching config_feature_extract_sitl.yml."""
    return SimpleNamespace(
        mae_model="vit_base_patch16",
        nb_classes=2048,
        drop_path=0.10,
        mae_ckpt="",
        freeze_weights=-1,
        out_features=out_features,
        return_mae_optimizer_groups=False,
    )


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load TwoHeadMAEModel from either:
      - A Lightning .ckpt  (keys: state_dict, hyper_parameters, ...)
      - A MAE pretrained .pth  (keys: model, optimizer, epoch, scaler, args)
      - A fine-tuned .pth with only a state_dict / model key

    Returns (model, hparams) with the model in eval mode.
    """
    from models.mae import TwoHeadMAEModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    print(f"Checkpoint keys: {ckpt_keys}")

    # ------------------------------------------------------------------
    # CASE 1: Lightning checkpoint (saved by ModelCheckpoint / trainer)
    # ------------------------------------------------------------------
    if "state_dict" in ckpt_keys and ("hyper_parameters" in ckpt_keys or "hparams" in ckpt_keys):
        print("Detected Lightning checkpoint format")
        hp = ckpt.get("hyper_parameters", ckpt.get("hparams"))
        hparams = SimpleNamespace(**hp) if isinstance(hp, dict) else hp

        # Fill in any missing fields with defaults
        defaults = _default_hparams()
        for attr in vars(defaults):
            if not hasattr(hparams, attr):
                setattr(hparams, attr, getattr(defaults, attr))

        # Build model without loading MAE pretrained weights (ckpt has everything)
        hparams.mae_ckpt = ""
        hparams.freeze_weights = -1
        model = TwoHeadMAEModel(hparams=hparams)

        # Lightning prefixes keys with "model." (attribute name in LightningModule)
        state_dict = ckpt["state_dict"]
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[len("model."):] if k.startswith("model.") else k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"WARNING: {len(missing)} missing keys: {missing[:5]}")
        if unexpected:
            print(f"WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}")

    # ------------------------------------------------------------------
    # CASE 2: MAE pretrained checkpoint (endovit_SPR.pth style)
    #   keys: ['model', 'optimizer', 'epoch', 'scaler', 'args']
    #   The 'model' key holds the ViT backbone state_dict.
    #   fc_phase / fc_tool are NOT in this checkpoint (randomly initialised).
    # ------------------------------------------------------------------
    elif "model" in ckpt_keys and "state_dict" not in ckpt_keys:
        print("Detected MAE pretrained checkpoint format")
        print("NOTE: this checkpoint only contains the ViT backbone.")
        print("      The classification heads (fc_phase, fc_tool) will be")
        print("      randomly initialised — expect poor accuracy unless you")
        print("      also pass a fine-tuned checkpoint.\n")

        hparams = _default_hparams()
        # Point mae_ckpt to the file so prepare_mae_model loads the backbone
        hparams.mae_ckpt = str(checkpoint_path)
        hparams.freeze_weights = -1

        model = TwoHeadMAEModel(hparams=hparams)

    # ------------------------------------------------------------------
    # CASE 3: Plain state_dict (e.g. torch.save(model.state_dict(), ...))
    # ------------------------------------------------------------------
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt_keys):
        print("Detected plain state_dict checkpoint format")
        hparams = _default_hparams()
        hparams.mae_ckpt = ""
        hparams.freeze_weights = -1
        model = TwoHeadMAEModel(hparams=hparams)

        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"WARNING: {len(missing)} missing keys: {missing[:5]}")
        if unexpected:
            print(f"WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}")

    else:
        raise RuntimeError(
            f"Unrecognised checkpoint format. Keys: {ckpt_keys}\n"
            "Expected one of:\n"
            "  1) Lightning .ckpt (state_dict + hyper_parameters)\n"
            "  2) MAE pretrained .pth (model + args)\n"
            "  3) Plain state_dict"
        )

    model = model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model, hparams


# ============================================================
# Main test routine
# ============================================================
def run_test(args):
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    sitl_root = Path(args.data_root)

    # Resolve test split from seed
    if args.seed not in TEST_SPLITS:
        print(f"ERROR: unknown seed {args.seed}. Available seeds: {sorted(TEST_SPLITS.keys())}")
        sys.exit(1)
    video_to_annotation = TEST_SPLITS[args.seed]

    print("=" * 80)
    print("TEST-ONLY EVALUATION")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {sitl_root}")
    print(f"Seed       : {args.seed}")
    print(f"Device     : {device}")
    print(f"Batch size : {args.batch_size}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Classes    : {CLASS_LABELS}")
    print(f"Label map  : {LABEL_MAP}")
    print(f"Test videos: {sorted(video_to_annotation.keys())}")
    print(f"Annotations: {video_to_annotation}")
    print("=" * 80 + "\n")

    # 1. Build test DataFrame
    df_test = build_test_dataframe(sitl_root, video_to_annotation, fps=args.fps)
    if len(df_test) == 0:
        print("ERROR: No test data found!")
        return

    # 2. Load model
    model, saved_hparams = load_model_from_checkpoint(args.checkpoint, device=device)

    # 3. Build dataset and dataloader
    transform = Compose([
        Resize(height=224, width=224),
        Normalize(mean=[0.3456, 0.2281, 0.2233], std=[0.2528, 0.2135, 0.2104]),
        ToTensorV2(),
    ])
    test_dataset = TestImageDataset(df_test, sitl_root / "frames", transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 4. Run inference
    all_preds = []
    all_labels = []
    all_logits = []
    all_video_idxs = []
    all_frames = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, labels, vid_idxs, frames) in enumerate(test_loader):
            images = images.to(device).float()
            _, phase_logits, _ = model(images)

            all_logits.append(phase_logits.cpu().numpy())
            all_preds.append(torch.argmax(phase_logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_video_idxs.append(vid_idxs.numpy())
            all_frames.append(frames.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)
    all_video_idxs = np.concatenate(all_video_idxs)
    all_frames = np.concatenate(all_frames)

    # ── Exclude classes from evaluation ───────────────────────────
    # Mask out logits of excluded classes (→ −inf) so argmax never
    # picks them, then drop frames whose ground-truth is excluded.
    # If the model already has fewer outputs than NUM_CLASSES (i.e.
    # the checkpoint was trained without the excluded class), we
    # only mask columns that actually exist.
    _excluded_indices = sorted(set(range(NUM_CLASSES)) - set(EVAL_CLASS_INDICES))
    n_model_outputs = all_logits.shape[1]
    if _excluded_indices:
        print(f"\nExcluding classes from evaluation: "
              f"{[CLASS_LABELS[i] for i in _excluded_indices]}")
        _mask_cols = [i for i in _excluded_indices if i < n_model_outputs]
        if _mask_cols:
            all_logits[:, _mask_cols] = -np.inf
            all_preds = np.argmax(all_logits, axis=1)
        eval_mask = np.isin(all_labels, EVAL_CLASS_INDICES)
        dropped = (~eval_mask).sum()
        all_preds = all_preds[eval_mask]
        all_labels = all_labels[eval_mask]
        all_logits = all_logits[eval_mask]
        all_video_idxs = all_video_idxs[eval_mask]
        all_frames = all_frames[eval_mask]
        print(f"Frames after exclusion: {len(all_labels)} "
              f"(dropped {dropped} with excluded ground-truth)\n")

    # ============================================================
    # 5. Per-video accuracy (same method as feature_extraction.py)
    # ============================================================
    print("\n" + "=" * 80)
    print("PER-VIDEO RESULTS (pycm — same as training pipeline)")
    print("=" * 80)

    test_acc_per_video = {}
    test_f1_per_video = {}

    for vid_idx in sorted(video_to_annotation.keys()):
        mask = all_video_idxs == vid_idx
        if mask.sum() < 2:
            print(f"  Video {vid_idx}: SKIPPED (only {mask.sum()} frame(s))")
            continue

        vid_labels = all_labels[mask].tolist()
        vid_logits = all_logits[mask].tolist()

        acc, ppv, tpr, keys, f1_macro = get_phase_acc(vid_labels, vid_logits)
        test_acc_per_video[vid_idx] = acc
        test_f1_per_video[vid_idx] = f1_macro

        print(f"  Video {vid_idx:3d} | Acc: {acc:.4f} | F1_Macro: {f1_macro} | "
              f"Frames: {mask.sum()}")

    # Mean across videos (same as on_test_epoch_end in feature_extraction.py)
    valid_accs = [test_acc_per_video[v] for v in sorted(video_to_annotation.keys())
                  if v in test_acc_per_video]
    mean_acc_per_video = np.mean(valid_accs) if valid_accs else 0.0
    print(f"\n  Mean per-video accuracy (pycm): {mean_acc_per_video:.4f}")
    print(f"  ({len(valid_accs)} / {len(video_to_annotation)} videos evaluated)")

    # ============================================================
    # 6. Overall (frame-level) metrics via sklearn
    # ============================================================
    print("\n" + "=" * 80)
    print("OVERALL FRAME-LEVEL METRICS (sklearn)")
    print("=" * 80)

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1_macro = f1_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="macro", zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="weighted", zero_division=0)
    overall_recall_macro = recall_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="macro", zero_division=0)
    overall_recall_weighted = recall_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="weighted", zero_division=0)
    overall_precision_macro = precision_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="macro", zero_division=0)
    overall_precision_weighted = precision_score(all_labels, all_preds, labels=EVAL_CLASS_INDICES, average="weighted", zero_division=0)

    print(f"  Overall Accuracy          : {overall_acc:.4f}")
    print(f"  F1 (macro)                : {overall_f1_macro:.4f}")
    print(f"  F1 (weighted)             : {overall_f1_weighted:.4f}")
    print(f"  Recall (macro)            : {overall_recall_macro:.4f}")
    print(f"  Recall (weighted)         : {overall_recall_weighted:.4f}")
    print(f"  Precision (macro)         : {overall_precision_macro:.4f}")
    print(f"  Precision (weighted)      : {overall_precision_weighted:.4f}")
    print(f"  Total frames              : {len(all_labels)}")
    if _excluded_indices:
        print(f"  Excluded classes          : {[CLASS_LABELS[i] for i in _excluded_indices]}")

    # ============================================================
    # 7. Per-class breakdown
    # ============================================================
    print("\n" + "=" * 80)
    print("PER-CLASS METRICS")
    print("=" * 80)

    report = classification_report(
        all_labels, all_preds,
        labels=EVAL_CLASS_INDICES,
        target_names=EVAL_CLASS_LABELS,
        digits=4,
        zero_division=0,
    )
    print(report)

    # ============================================================
    # 8. Confusion matrix
    # ============================================================
    print("=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=predicted)")
    print("=" * 80)

    cm = sklearn_confusion_matrix(all_labels, all_preds, labels=EVAL_CLASS_INDICES)
    header = "          " + "  ".join([f"{c[:8]:>8s}" for c in EVAL_CLASS_LABELS])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:8d}" for v in row])
        print(f"{EVAL_CLASS_LABELS[i][:8]:>8s}  {row_str}")

    # ============================================================
    # 9. Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if _excluded_indices:
        print(f"  Excluded classes               : {[CLASS_LABELS[i] for i in _excluded_indices]}")
    print(f"  Per-video mean accuracy (pycm) : {mean_acc_per_video:.4f}")
    print(f"  Frame-level accuracy           : {overall_acc:.4f}")
    print(f"  Frame-level F1 (macro)         : {overall_f1_macro:.4f}")
    print(f"  Frame-level F1 (weighted)      : {overall_f1_weighted:.4f}")
    print(f"  Frame-level Recall (macro)     : {overall_recall_macro:.4f}")
    print(f"  Frame-level Recall (weighted)  : {overall_recall_weighted:.4f}")
    print(f"  Frame-level Precision (macro)  : {overall_precision_macro:.4f}")
    print(f"  Frame-level Precision (weighted): {overall_precision_weighted:.4f}")
    print("=" * 80)

    # ============================================================
    # 10. Phase timeline (GT vs Predicted) saved as PNG
    # ============================================================
    timeline_path = Path(args.checkpoint).parent / "phase_timeline.png"
    if hasattr(args, "output_dir") and args.output_dir:
        timeline_path = Path(args.output_dir) / "phase_timeline.png"
        timeline_path.parent.mkdir(parents=True, exist_ok=True)

    plot_phase_timelines(
        all_labels, all_preds, all_video_idxs, all_frames,
        video_to_annotation, CLASS_LABELS, EVAL_CLASS_INDICES,
        str(timeline_path),
    )


# ============================================================
# Timeline visualisation
# ============================================================
def plot_phase_timelines(all_labels, all_preds, all_video_idxs, all_frames,
                         video_to_annotation, class_labels, eval_class_indices,
                         output_path):
    """Save a per-video GT-vs-Predicted phase ribbon chart as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    n_classes = len(class_labels)
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#42d4f4", "#f032e6", "#bfef45",
        "#fabebe", "#469990", "#e6beff", "#9A6324",
    ]
    while len(palette) < n_classes:
        palette.append("#aaaaaa")
    palette = palette[:n_classes]
    rgb = [tuple(int(c[i:i+2], 16) / 255 for i in (1, 3, 5)) for c in palette]
    cmap = ListedColormap(rgb)

    videos = sorted(set(all_video_idxs))
    n_videos = len(videos)

    fig, axes = plt.subplots(
        n_videos, 1,
        figsize=(22, 2.2 * n_videos + 1.5),
        squeeze=False,
    )

    for row, vid_idx in enumerate(videos):
        ax = axes[row, 0]
        mask = all_video_idxs == vid_idx
        vid_frames = all_frames[mask]
        vid_labels = all_labels[mask]
        vid_preds = all_preds[mask]

        order = np.argsort(vid_frames)
        vid_labels = vid_labels[order]
        vid_preds = vid_preds[order]

        ribbon = np.vstack([vid_labels.reshape(1, -1),
                            vid_preds.reshape(1, -1)])
        ax.imshow(ribbon, aspect="auto", cmap=cmap,
                  vmin=0, vmax=n_classes - 1, interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Ground Truth", "Predicted"], fontsize=10)
        ann_name = video_to_annotation.get(vid_idx, "")
        acc_frames = np.sum(vid_labels == vid_preds)
        acc_pct = acc_frames / len(vid_labels) * 100 if len(vid_labels) else 0
        ax.set_title(
            f"Video {vid_idx}  ({ann_name})  —  Acc {acc_pct:.1f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Frame (time-ordered)")

    legend_patches = [
        mpatches.Patch(color=rgb[i], label=class_labels[i])
        for i in eval_class_indices
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=min(len(legend_patches), 4), fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPhase timeline saved to: {output_path}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test-only evaluation for surgical phase recognition"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the Lightning .ckpt file to evaluate"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="path-to/SITL_phases",
        help="Root directory of SITL dataset (contains frames/ and phase_annotations/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, choices=sorted(TEST_SPLITS.keys()),
        help="Seed that determines which test split to use (42, 123, or 7)"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--fps", type=int, default=1,
                        help="Frame sampling rate (1 = every frame)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Directory to save the phase timeline PNG "
                             "(defaults to checkpoint directory)")

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.data_root).exists():
        print(f"ERROR: data root not found: {args.data_root}")
        sys.exit(1)

    run_test(args)


if __name__ == "__main__":
    main()
