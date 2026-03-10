#!/usr/bin/env python3
"""
Simple test-only evaluation script for Cholec80.

It loads a checkpoint, runs inference on frames from selected test videos,
and compares predictions against labels from phase annotations.

Reviewer note:
This script is the most convenient reviewer-side entrypoint for inspecting
laparoscopic evaluation because it works directly from frame folders and phase
annotation files, without requiring the full training pipeline.

Split preset → test videos:
  split=1: 49-80
  split=2: 1-8 + 57-80
  split=3: 1-16 + 65-80

Example:
  python test_only_cholec80.py \
      --checkpoint /path/to/model.ckpt \
      --split_id 1
"""

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Paths for model imports (works from any cwd). The release copy keeps the
# original import structure so the model definition stays identical to the
# training code used in the paper.
# ---------------------------------------------------------------------------
SCRIPT_ROOT = Path(__file__).resolve().parent
TECNO_ROOT = SCRIPT_ROOT / "finetuning" / "surgical_phase_recognition" / "model" / "TeCNO"
MAE_ROOT = SCRIPT_ROOT / "pretraining" / "mae"
sys.path.insert(0, str(TECNO_ROOT))
sys.path.insert(0, str(MAE_ROOT))


# ---------------------------------------------------------------------------
# Cholec80 labels and split mapping
# ---------------------------------------------------------------------------
CLASS_LABELS = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    # "CleaningCoagulation",
    # "GallbladderRetraction",
]
PHASE_TO_LABEL = {name.lower(): idx for idx, name in enumerate(CLASS_LABELS)}
NUM_CLASSES = len(CLASS_LABELS)

SPLIT_TO_TEST_VIDEOS = {
    1: [i for i in range(49, 81)],
    2: [i for i in range(1, 9)] + [i for i in range(57, 81)],
    3: [i for i in range(1, 17)] + [i for i in range(65, 81)],
}


def build_video_to_annotation(videos):
    return {v: f"video{v:02d}-phase.txt" for v in videos}


def infer_split_id_from_checkpoint(checkpoint_path: Path):
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"WARNING: could not read checkpoint metadata: {exc}")
        return None

    if not isinstance(ckpt, dict):
        return None

    hp = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
    if isinstance(hp, SimpleNamespace):
        split_id = getattr(hp, "video_split_id", None)
    elif isinstance(hp, dict):
        split_id = hp.get("video_split_id", None)
    else:
        split_id = None

    if split_id is None:
        return None

    try:
        split_id = int(split_id)
    except (TypeError, ValueError):
        return None

    return split_id if split_id in SPLIT_TO_TEST_VIDEOS else None


def _read_cholec_pickle_compat(pickle_path: Path):
    # Compatibility shim for pickles produced in environments where
    # `numpy._core.numeric` was importable.
    import numpy.core.numeric as np_core_numeric

    sys.modules.setdefault("numpy._core.numeric", np_core_numeric)
    return pd.read_pickle(pickle_path)


def build_eval_dataframe_from_pickle(data_root: Path, test_videos, fps: int = 1):
    pkl_path = data_root / "cholec_annotations_1fps.pkl"
    if not pkl_path.exists():
        return None

    try:
        df_all = _read_cholec_pickle_compat(pkl_path)
    except Exception as exc:
        print(f"WARNING: failed to load {pkl_path}: {exc}")
        return None

    required_cols = {"image_path", "class", "video_idx"}
    if not required_cols.issubset(df_all.columns):
        print(
            f"WARNING: {pkl_path} missing required columns "
            f"{sorted(required_cols - set(df_all.columns))}."
        )
        return None

    df = df_all[df_all["video_idx"].isin(test_videos)].copy()
    if df.empty:
        print(f"WARNING: no rows in {pkl_path} for videos {test_videos}.")
        return None

    df["video_idx"] = df["video_idx"].astype(int)
    df["class"] = df["class"].astype(int)
    df["frame"] = df["image_path"].map(lambda p: extract_frame_number(Path(str(p))))
    df = df.dropna(subset=["frame"])
    df["frame"] = df["frame"].astype(int)
    df = df[(df["class"] >= 0) & (df["class"] < NUM_CLASSES)].copy()

    df = df.sort_values(["video_idx", "frame", "image_path"])
    if fps > 1:
        keep_mask = (df.groupby("video_idx").cumcount() % fps) == 0
        df = df[keep_mask].reset_index(drop=True)

    print("=" * 80)
    print("EVAL DATA SUMMARY (from cholec_annotations_1fps.pkl)")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Videos: {sorted(df['video_idx'].unique().tolist())}")
    print("Class distribution:")
    print(df["class"].value_counts().sort_index())
    print("=" * 80)
    return df[["image_path", "class", "video_idx", "frame"]]


def phase_to_label(phase_name):
    if phase_name is None or pd.isna(phase_name):
        return None
    key = str(phase_name).strip().lower()
    if not key:
        return None
    return PHASE_TO_LABEL.get(key)


def extract_frame_number(frame_path: Path):
    stem = frame_path.stem
    if "_" in stem:
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def nearest_annotation_frame(ann_frames: np.ndarray, target: int):
    idx = np.searchsorted(ann_frames, target)
    candidates = []
    if idx < len(ann_frames):
        candidates.append(int(ann_frames[idx]))
    if idx > 0:
        candidates.append(int(ann_frames[idx - 1]))
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - target))


def build_eval_dataframe(data_root: Path, video_to_annotation: dict, fps: int = 1):
    """
    Build dataframe with columns: image_path, class, video_idx, frame.

    - image_path is relative to data_root/frames
    - class is integer in [0, 6]
    """
    frames_root = data_root / "frames"
    ann_root = data_root / "phase_annotations"

    rows = []
    skipped_missing = 0
    skipped_unknown = 0

    for video_idx, ann_name in sorted(video_to_annotation.items()):
        video_dir = frames_root / f"video{video_idx:02d}"
        ann_path = ann_root / ann_name

        if not video_dir.exists():
            print(f"WARNING: missing video directory: {video_dir}")
            continue
        if not ann_path.exists():
            print(f"WARNING: missing annotation file: {ann_path}")
            continue

        ann_df = pd.read_csv(ann_path, sep="\t", dtype={"Frame": int, "Phase": str})
        ann_df = ann_df.dropna(subset=["Phase"]).copy()
        ann_df["Phase"] = ann_df["Phase"].astype(str).str.strip()
        ann_df = ann_df[ann_df["Phase"].str.len() > 0]
        if ann_df.empty:
            print(f"WARNING: no valid labels in {ann_path}")
            continue

        frame_to_phase = dict(zip(ann_df["Frame"].astype(int), ann_df["Phase"]))
        ann_frames = np.array(sorted(frame_to_phase.keys()), dtype=np.int64)
        max_ann_frame = int(ann_frames[-1])

        frame_files = sorted(
            [
                p
                for p in video_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        if not frame_files:
            print(f"WARNING: no image frames found in {video_dir}")
            continue

        if fps > 1:
            frame_files = frame_files[::fps]

        parsed = []
        for frame_path in frame_files:
            frame_num = extract_frame_number(frame_path)
            if frame_num is not None:
                parsed.append((frame_path, frame_num))

        if not parsed:
            print(f"WARNING: could not parse frame numbers in {video_dir}")
            continue

        disk_numbers = np.array([x[1] for x in parsed], dtype=np.int64)
        min_disk = int(disk_numbers.min())
        max_disk = int(disk_numbers.max())
        disk_span = max_disk - min_disk

        # Cholec80 annotations are at 25 fps while frames here are 1 fps.
        scale = (max_ann_frame / disk_span) if (disk_span > 0 and max_ann_frame > 0) else 1.0

        for frame_path, frame_num in parsed:
            ann_frame = int(round((frame_num - min_disk) * scale))
            ann_frame = max(0, min(ann_frame, max_ann_frame))

            if ann_frame in frame_to_phase:
                phase_name = frame_to_phase[ann_frame]
            else:
                nearest = nearest_annotation_frame(ann_frames, ann_frame)
                if nearest is None:
                    skipped_missing += 1
                    continue
                phase_name = frame_to_phase.get(nearest)

            label = phase_to_label(phase_name)
            if label is None:
                skipped_unknown += 1
                continue

            rows.append(
                {
                    "image_path": f"video{video_idx:02d}/{frame_path.name}",
                    "class": int(label),
                    "video_idx": int(video_idx),
                    "frame": int(frame_num),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No evaluation rows built.")
        return df

    print("=" * 80)
    print("EVAL DATA SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Videos: {sorted(df['video_idx'].unique().tolist())}")
    print(f"Skipped missing label match: {skipped_missing}")
    print(f"Skipped unknown labels: {skipped_unknown}")
    print("Class distribution:")
    print(df["class"].value_counts().sort_index())
    print("=" * 80)

    return df


class Cholec80FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, frames_root: Path, transform):
        self.df = df.reset_index(drop=True)
        self.frames_root = frames_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.frames_root / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return (
            image,
            int(row["class"]),
            int(row["video_idx"]),
            int(row["frame"]),
        )


def _default_hparams(out_features=NUM_CLASSES):
    return SimpleNamespace(
        mae_model="vit_base_patch16",
        nb_classes=2048,
        drop_path=0.10,
        mae_ckpt="",
        freeze_weights=-1,
        out_features=out_features,
        return_mae_optimizer_groups=False,
        mae_reinit_n_layers=-1,
        mae_weight_decay=0.0,
        mae_layer_decay=1.0,
    )


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[len("model.") :] if key.startswith("model.") else key
        cleaned[new_key] = value
    return cleaned


def _ensure_torchsummary_stub():
    # Some environments do not have torchsummary installed, but TeCNO's
    # model file imports it unconditionally. Stub it to avoid import failure.
    if "torchsummary" not in sys.modules:
        stub = types.ModuleType("torchsummary")
        stub.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = stub


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    _ensure_torchsummary_stub()
    from models.mae import TwoHeadMAEModel

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    ckpt_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    print(f"Checkpoint keys: {ckpt_keys}")

    if "state_dict" in ckpt_keys:
        hp = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
        hparams = SimpleNamespace(**hp) if isinstance(hp, dict) else hp
        if not isinstance(hparams, SimpleNamespace):
            hparams = SimpleNamespace()

        defaults = _default_hparams()
        for attr in vars(defaults):
            if not hasattr(hparams, attr):
                setattr(hparams, attr, getattr(defaults, attr))

        hparams.mae_ckpt = ""
        hparams.freeze_weights = -1

        model = TwoHeadMAEModel(hparams=hparams)
        state_dict = _clean_state_dict_keys(ckpt["state_dict"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    elif "model" in ckpt_keys and isinstance(ckpt["model"], dict):
        inner_state = ckpt["model"]
        inner_keys = list(inner_state.keys())
        has_phase_head = any("fc_phase" in key for key in inner_keys)

        if has_phase_head:
            hparams = _default_hparams()
            model = TwoHeadMAEModel(hparams=hparams)
            state_dict = _clean_state_dict_keys(inner_state)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            print(
                "Detected MAE-pretrained checkpoint (backbone only). "
                "Phase head will be random unless you use a fine-tuned checkpoint."
            )
            hparams = _default_hparams()
            hparams.mae_ckpt = str(checkpoint_path)
            hparams.freeze_weights = -1
            model = TwoHeadMAEModel(hparams=hparams)
            missing, unexpected = [], []

    elif isinstance(ckpt, dict):
        hparams = _default_hparams()
        model = TwoHeadMAEModel(hparams=hparams)
        state_dict = _clean_state_dict_keys(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    else:
        raise RuntimeError(
            f"Unsupported checkpoint format at {checkpoint_path}. Keys: {ckpt_keys}"
        )

    if missing:
        print(f"WARNING: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"WARNING: unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    return model


def run_test(args):
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    checkpoint_path = Path(args.checkpoint)
    ckpt_split_id = infer_split_id_from_checkpoint(checkpoint_path)
    split_id = args.split_id if args.split_id is not None else ckpt_split_id
    if split_id is None:
        split_id = 1

    if split_id not in SPLIT_TO_TEST_VIDEOS:
        raise ValueError(
            f"Unknown split_id {split_id}. Available: {sorted(SPLIT_TO_TEST_VIDEOS.keys())}"
        )

    if ckpt_split_id is not None and args.split_id is not None and args.split_id != ckpt_split_id:
        print(
            f"WARNING: split_id={args.split_id} differs from checkpoint video_split_id={ckpt_split_id}. "
            "Using the explicit CLI split_id."
        )

    test_videos = SPLIT_TO_TEST_VIDEOS[split_id]
    video_to_annotation = build_video_to_annotation(test_videos)
    data_root = Path(args.data_root)

    print("=" * 80)
    print("CHOLEC80 TEST-ONLY EVALUATION")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {data_root}")
    print(f"Split ID   : {split_id}")
    print(f"Ckpt split : {ckpt_split_id if ckpt_split_id is not None else 'N/A'}")
    print(f"Test videos: {test_videos}")
    print(f"Device     : {device}")
    print("=" * 80)

    df = build_eval_dataframe_from_pickle(data_root, test_videos, fps=args.fps)
    if df is None:
        df = build_eval_dataframe(data_root, video_to_annotation, fps=args.fps)
    if df.empty:
        print("ERROR: no samples found for evaluation.")
        return

    model = load_model_from_checkpoint(checkpoint_path, device=device)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.3456, 0.2281, 0.2233], std=[0.2528, 0.2135, 0.2104]),
        ]
    )

    dataset = Cholec80FrameDataset(df, data_root / "frames", transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_logits = []
    all_labels = []
    all_video_idxs = []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (images, labels, vid_idxs, _frames) in enumerate(loader):
            images = images.to(device).float()
            _, phase_logits, _ = model(images)
            all_logits.append(phase_logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_video_idxs.append(vid_idxs.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}")

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    video_idxs = np.concatenate(all_video_idxs)

    if logits.shape[1] > NUM_CLASSES:
        print(
            f"WARNING: model outputs {logits.shape[1]} classes; trimming to {NUM_CLASSES}."
        )
        logits = logits[:, :NUM_CLASSES]
    elif logits.shape[1] < NUM_CLASSES:
        print(
            f"WARNING: model outputs only {logits.shape[1]} classes "
            f"(expected {NUM_CLASSES})."
        )

    preds = np.argmax(logits, axis=1)

    print("\n" + "=" * 80)
    print("PER-VIDEO ACCURACY")
    print("=" * 80)
    per_video_acc = {}
    for vid in test_videos:
        mask = video_idxs == vid
        if mask.sum() == 0:
            print(f"Video {vid:02d}: no samples")
            continue
        acc = accuracy_score(labels[mask], preds[mask])
        per_video_acc[vid] = acc
        print(f"Video {vid:02d}: acc={acc:.4f}  frames={mask.sum()}")

    mean_video_acc = float(np.mean(list(per_video_acc.values()))) if per_video_acc else 0.0

    eval_labels = list(range(NUM_CLASSES))
    overall_acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, labels=eval_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(
        labels,
        preds,
        labels=eval_labels,
        average="weighted",
        zero_division=0,
    )
    precision_macro = precision_score(
        labels,
        preds,
        labels=eval_labels,
        average="macro",
        zero_division=0,
    )
    recall_macro = recall_score(
        labels,
        preds,
        labels=eval_labels,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"Accuracy             : {overall_acc:.4f}")
    print(f"F1 (macro)           : {f1_macro:.4f}")
    print(f"F1 (weighted)        : {f1_weighted:.4f}")
    print(f"Precision (macro)    : {precision_macro:.4f}")
    print(f"Recall (macro)       : {recall_macro:.4f}")
    print(f"Per-video Accuracy   : {mean_video_acc:.4f}")

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(
        classification_report(
            labels,
            preds,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("=" * 80)
    print("CONFUSION MATRIX (rows=true, cols=pred)")
    print("=" * 80)
    header = " " * 18 + " ".join([f"{name[:8]:>8s}" for name in CLASS_LABELS])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{v:8d}" for v in row])
        print(f"{CLASS_LABELS[i][:8]:>8s}  {row_str}")


def main():
    parser = argparse.ArgumentParser(description="Test-only evaluation on Cholec80")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data_root",
        type=str,
        default="path-to/cholec80",
        help="Cholec80 root folder with frames/ and phase_annotations/",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        choices=sorted(SPLIT_TO_TEST_VIDEOS.keys()),
        help=(
            "Cholec80 split preset ID (1/2/3). "
            "If omitted, uses checkpoint hyper_parameters.video_split_id when available."
        ),
    )
    # Backward-compatible alias for older runs using --seed as split ID.
    parser.add_argument(
        "--seed",
        dest="split_id",
        type=int,
        choices=sorted(SPLIT_TO_TEST_VIDEOS.keys()),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frame sampling stride on the extracted frame list (1 = use all)",
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    run_test(args)


if __name__ == "__main__":
    main()
