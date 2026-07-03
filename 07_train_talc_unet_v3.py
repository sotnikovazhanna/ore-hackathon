from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.talc_unet import TalcUNetResNet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not str(path).strip() or str(path).lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def pad_to_size(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h or pad_w:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return rgb, mask


def random_crop(rgb: np.ndarray, mask: np.ndarray, size: int, positive_probability: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    rgb, mask = pad_to_size(rgb, mask, size)
    h, w = rgb.shape[:2]

    if mask.any() and random.random() < positive_probability:
        ys, xs = np.where(mask > 0)
        chosen = random.randrange(len(xs))
        center_x, center_y = int(xs[chosen]), int(ys[chosen])
        jitter = size // 4
        center_x += random.randint(-jitter, jitter)
        center_y += random.randint(-jitter, jitter)
        x0 = int(np.clip(center_x - size // 2, 0, w - size))
        y0 = int(np.clip(center_y - size // 2, 0, h - size))
    else:
        x0 = random.randint(0, w - size)
        y0 = random.randint(0, h - size)

    return rgb[y0 : y0 + size, x0 : x0 + size], mask[y0 : y0 + size, x0 : x0 + size]


def augment(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        rgb, mask = np.fliplr(rgb), np.fliplr(mask)
    if random.random() < 0.5:
        rgb, mask = np.flipud(rgb), np.flipud(mask)

    k = random.randint(0, 3)
    rgb, mask = np.rot90(rgb, k), np.rot90(mask, k)

    image = Image.fromarray(np.ascontiguousarray(rgb))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.35:
        gamma = random.uniform(0.70, 1.35)
        array = np.asarray(image).astype(np.float32) / 255.0
        array = np.power(np.clip(array, 0, 1), gamma)
        image = Image.fromarray((array * 255).astype(np.uint8))

    if random.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    rgb = np.asarray(image)
    if random.random() < 0.20:
        noise = np.random.normal(0, random.uniform(2, 8), size=rgb.shape).astype(np.float32)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)


def to_tensor(rgb: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
    mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)
    return image_tensor, mask_tensor


class TalcDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, crop_size: int, repeats_positive: int, repeats_negative: int) -> None:
        self.crop_size = crop_size
        self.records: list[pd.Series] = []
        for _, row in frame.iterrows():
            repeats = repeats_positive if int(row["target_presence"]) == 1 else repeats_negative
            self.records.extend([row] * repeats)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        rgb = load_rgb(row["image_path"])
        mask = load_mask(row.get("mask_path", ""), rgb.shape[:2])
        rgb, mask = random_crop(rgb, mask, self.crop_size)
        rgb, mask = augment(rgb, mask)
        image_tensor, mask_tensor = to_tensor(rgb, mask)
        return {"image": image_tensor, "mask": mask_tensor}


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """Pixel loss + overlap loss + a small area-consistency term.

    The manual masks are polygonal and the applied decision rule depends on the
    predicted area (>10%), so the area term helps the network match the total
    amount of talc without overfitting every boundary pixel.
    """
    probability = torch.sigmoid(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    overlap = dice_loss(logits, target)
    predicted_area = probability.mean(dim=(1, 2, 3))
    target_area = target.mean(dim=(1, 2, 3))
    area_loss = torch.abs(predicted_area - target_area).mean()
    return bce + 0.8 * overlap + 0.25 * area_loss


def positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    result = list(range(0, length - tile_size + 1, stride))
    if result[-1] != length - tile_size:
        result.append(length - tile_size)
    return result


def segment_probability(model: nn.Module, rgb: np.ndarray, device: torch.device, tile_size: int, overlap: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    stride = tile_size - overlap
    accumulator = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    model.eval()
    for y in positions(h, tile_size, stride):
        for x in positions(w, tile_size, stride):
            tile = rgb[y : y + tile_size, x : x + tile_size]
            valid_h, valid_w = tile.shape[:2]
            if valid_h < tile_size or valid_w < tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    tile_size - valid_h,
                    0,
                    tile_size - valid_w,
                    cv2.BORDER_REFLECT_101,
                )
            tensor, _ = to_tensor(tile, np.zeros(tile.shape[:2], dtype=np.uint8))
            with torch.no_grad():
                probability = torch.sigmoid(model(tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            probability = probability[:valid_h, :valid_w]
            accumulator[y : y + valid_h, x : x + valid_w] += probability
            weights[y : y + valid_h, x : x + valid_w] += 1

    return accumulator / np.maximum(weights, 1)


def postprocess(probability: np.ndarray, threshold: float) -> np.ndarray:
    mask = (probability >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    min_area = max(25, int(mask.size * 0.00035))
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            result[labels == component] = 1
    return result


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return (mask.astype(np.uint8) - eroded).astype(bool)


def hausdorff_metrics(true_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float]:
    true_boundary = boundary(true_mask)
    pred_boundary = boundary(pred_mask)
    diagonal = math.hypot(*true_mask.shape)
    if not true_boundary.any() and not pred_boundary.any():
        return 0.0, 0.0
    if not true_boundary.any() or not pred_boundary.any():
        return diagonal, diagonal

    distance_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_true = cv2.distanceTransform((~true_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([distance_to_pred[true_boundary], distance_to_true[pred_boundary]])
    return float(distances.max()), float(np.percentile(distances, 95))


def iou_and_dice(true_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float]:
    true_bool, pred_bool = true_mask.astype(bool), pred_mask.astype(bool)
    intersection = np.logical_and(true_bool, pred_bool).sum()
    union = np.logical_or(true_bool, pred_bool).sum()
    iou = intersection / union if union else 1.0
    denom = true_bool.sum() + pred_bool.sum()
    dice = 2 * intersection / denom if denom else 1.0
    return float(iou), float(dice)


def _binary_metrics(true_values: np.ndarray, predicted_values: np.ndarray) -> dict[str, float]:
    true_values = true_values.astype(np.uint8)
    predicted_values = predicted_values.astype(np.uint8)
    tp = int(np.logical_and(true_values == 1, predicted_values == 1).sum())
    tn = int(np.logical_and(true_values == 0, predicted_values == 0).sum())
    fp = int(np.logical_and(true_values == 0, predicted_values == 1).sum())
    fn = int(np.logical_and(true_values == 1, predicted_values == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "threshold_accuracy": float(accuracy),
        "threshold_precision": float(precision),
        "threshold_recall": float(recall),
        "threshold_f1": float(f1),
        "threshold_tp": tp,
        "threshold_tn": tn,
        "threshold_fp": fp,
        "threshold_fn": fn,
    }


def evaluate(model: nn.Module, frame: pd.DataFrame, device: torch.device, threshold: float, tile_size: int, overlap: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for row in tqdm(frame.itertuples(), total=len(frame), desc=f"evaluate {threshold:.2f}", leave=False):
        rgb = load_rgb(row.image_path)
        true_mask = load_mask(row.mask_path, rgb.shape[:2])
        probability = segment_probability(model, rgb, device, tile_size, overlap)
        pred_mask = postprocess(probability, threshold)
        iou, dice = iou_and_dice(true_mask, pred_mask)
        hd, hd95 = hausdorff_metrics(true_mask, pred_mask)
        true_share = float(true_mask.mean() * 100)
        predicted_share = float(pred_mask.mean() * 100)
        rows.append(
            {
                "filename": row.filename,
                "target_presence": int(row.target_presence),
                "iou": iou,
                "dice": dice,
                "hausdorff_pixels": hd,
                "hausdorff95_pixels": hd95,
                "true_share_percent": true_share,
                "predicted_share_percent": predicted_share,
                "share_error_pp": abs(true_share - predicted_share),
                "true_over_10": int(true_share > 10.0),
                "predicted_over_10": int(predicted_share > 10.0),
            }
        )

    metrics = pd.DataFrame(rows)
    positives = metrics[metrics["target_presence"] == 1]
    negatives = metrics[metrics["target_presence"] == 0]
    threshold_metrics = _binary_metrics(
        metrics["true_over_10"].to_numpy(),
        metrics["predicted_over_10"].to_numpy(),
    )

    positive_mean_iou = float(positives["iou"].mean()) if len(positives) else 0.0
    positive_mean_dice = float(positives["dice"].mean()) if len(positives) else 0.0
    positive_area_mae = float(positives["share_error_pp"].mean()) if len(positives) else 0.0
    negative_predicted_share = (
        float(negatives["predicted_share_percent"].mean()) if len(negatives) else 0.0
    )
    negative_clean_rate = (
        float((negatives["predicted_share_percent"] < 1.0).mean()) if len(negatives) else 1.0
    )

    # Balanced score for checkpoint/threshold selection. It does not let empty
    # negative masks inflate IoU to 1.0 and it rewards the actual >10% decision.
    selection_score = (
        0.55 * positive_mean_iou
        + 0.30 * threshold_metrics["threshold_f1"]
        + 0.15 * negative_clean_rate
    )

    summary = {
        "images": len(metrics),
        "positive_images": len(positives),
        "negative_images": len(negatives),
        "positive_mean_iou": positive_mean_iou,
        "positive_median_iou": float(positives["iou"].median()) if len(positives) else 0.0,
        "positive_mean_dice": positive_mean_dice,
        "positive_mean_hausdorff_pixels": float(positives["hausdorff_pixels"].mean()) if len(positives) else 0.0,
        "positive_mean_hausdorff95_pixels": float(positives["hausdorff95_pixels"].mean()) if len(positives) else 0.0,
        "positive_area_mae_pp": positive_area_mae,
        "negative_mean_predicted_share_percent": negative_predicted_share,
        "negative_clean_rate_under_1_percent": negative_clean_rate,
        "overall_area_mae_pp": float(metrics["share_error_pp"].mean()),
        "selection_score": float(selection_score),
        **threshold_metrics,
    }
    return metrics, summary


def build_prediction_cache(
    model: nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    tile_size: int,
    overlap: int,
    description: str,
) -> list[dict]:
    """Run the neural network once per image and keep probability maps in RAM.

    Threshold calibration then reuses these maps instead of repeating expensive
    CPU inference for every candidate threshold.
    """
    cache: list[dict] = []
    for row in tqdm(frame.itertuples(), total=len(frame), desc=description):
        rgb = load_rgb(row.image_path)
        true_mask = load_mask(row.mask_path, rgb.shape[:2])
        probability = segment_probability(model, rgb, device, tile_size, overlap)
        cache.append(
            {
                "filename": row.filename,
                "target_presence": int(row.target_presence),
                "true_mask": true_mask,
                "probability": probability,
            }
        )
    return cache


def evaluate_cached(cache: list[dict], threshold: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    for item in cache:
        true_mask = item["true_mask"]
        pred_mask = postprocess(item["probability"], threshold)
        iou, dice = iou_and_dice(true_mask, pred_mask)
        hd, hd95 = hausdorff_metrics(true_mask, pred_mask)
        true_share = float(true_mask.mean() * 100)
        predicted_share = float(pred_mask.mean() * 100)
        rows.append(
            {
                "filename": item["filename"],
                "target_presence": item["target_presence"],
                "iou": iou,
                "dice": dice,
                "hausdorff_pixels": hd,
                "hausdorff95_pixels": hd95,
                "true_share_percent": true_share,
                "predicted_share_percent": predicted_share,
                "share_error_pp": abs(true_share - predicted_share),
                "true_over_10": int(true_share > 10.0),
                "predicted_over_10": int(predicted_share > 10.0),
            }
        )

    metrics = pd.DataFrame(rows)
    positives = metrics[metrics["target_presence"] == 1]
    negatives = metrics[metrics["target_presence"] == 0]
    threshold_metrics = _binary_metrics(
        metrics["true_over_10"].to_numpy(),
        metrics["predicted_over_10"].to_numpy(),
    )

    positive_mean_iou = float(positives["iou"].mean()) if len(positives) else 0.0
    positive_mean_dice = float(positives["dice"].mean()) if len(positives) else 0.0
    positive_area_mae = float(positives["share_error_pp"].mean()) if len(positives) else 0.0
    negative_predicted_share = (
        float(negatives["predicted_share_percent"].mean()) if len(negatives) else 0.0
    )
    negative_clean_rate = (
        float((negatives["predicted_share_percent"] < 1.0).mean()) if len(negatives) else 1.0
    )
    selection_score = (
        0.55 * positive_mean_iou
        + 0.30 * threshold_metrics["threshold_f1"]
        + 0.15 * negative_clean_rate
    )
    summary = {
        "images": len(metrics),
        "positive_images": len(positives),
        "negative_images": len(negatives),
        "positive_mean_iou": positive_mean_iou,
        "positive_median_iou": float(positives["iou"].median()) if len(positives) else 0.0,
        "positive_mean_dice": positive_mean_dice,
        "positive_mean_hausdorff_pixels": float(positives["hausdorff_pixels"].mean()) if len(positives) else 0.0,
        "positive_mean_hausdorff95_pixels": float(positives["hausdorff95_pixels"].mean()) if len(positives) else 0.0,
        "positive_area_mae_pp": positive_area_mae,
        "negative_mean_predicted_share_percent": negative_predicted_share,
        "negative_clean_rate_under_1_percent": negative_clean_rate,
        "overall_area_mae_pp": float(metrics["share_error_pp"].mean()),
        "selection_score": float(selection_score),
        **threshold_metrics,
    }
    return metrics, summary


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: AdamW, device: torch.device, pos_weight: torch.Tensor) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = combined_loss(logits, masks, pos_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train manual-mask talc U-Net.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manual_talc/talc_segmentation_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/talc_unet"))
    parser.add_argument("--model-path", type=Path, default=Path("models/talc_unet_resnet18.pth"))
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--positive-repeats", type=int, default=10)
    parser.add_argument("--negative-repeats", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    frame = pd.read_csv(args.manifest, encoding="utf-8-sig")
    for path in frame["image_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    for path in frame.loc[frame["target_presence"] == 1, "mask_path"]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    train_frame = frame[frame["split"] == "train"].copy()
    validation_frame = frame[frame["split"] == "validation"].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    print(frame.groupby(["split", "target_presence"]).size().unstack(fill_value=0))

    train_dataset = TalcDataset(
        train_frame,
        crop_size=args.crop_size,
        repeats_positive=args.positive_repeats,
        repeats_negative=args.negative_repeats,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TalcUNetResNet18(pretrained=True).to(device)
    model.freeze_encoder()
    pos_weight = torch.tensor([2.0], device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_score = -1.0
    best_state: dict | None = None

    stages = [
        ("head", args.head_epochs, 1e-3, False),
        ("finetune", args.finetune_epochs, 2e-4, True),
    ]

    epochs_without_improvement = 0
    for stage_name, epochs, learning_rate, unfreeze in stages:
        if unfreeze:
            model.unfreeze_encoder()
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            started = time.time()
            loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
            validation_metrics, validation_summary = evaluate(
                model,
                validation_frame,
                device,
                threshold=0.5,
                tile_size=args.crop_size,
                overlap=min(args.overlap, args.crop_size // 3),
            )
            row = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": loss,
                "validation_positive_mean_iou": validation_summary["positive_mean_iou"],
                "validation_positive_mean_dice": validation_summary["positive_mean_dice"],
                "validation_threshold_f1": validation_summary["threshold_f1"],
                "validation_negative_clean_rate": validation_summary["negative_clean_rate_under_1_percent"],
                "validation_selection_score": validation_summary["selection_score"],
                "seconds": time.time() - started,
            }
            history.append(row)
            print(
                f"[{stage_name}] {epoch}/{epochs} | loss={loss:.4f} | "
                f"val positive IoU={validation_summary['positive_mean_iou']:.4f} | "
                f"val >10 F1={validation_summary['threshold_f1']:.4f} | "
                f"score={validation_summary['selection_score']:.4f}"
            )

            if validation_summary["selection_score"] > best_score:
                best_score = validation_summary["selection_score"]
                epochs_without_improvement = 0
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                validation_metrics.to_csv(args.output_dir / "best_validation_metrics.csv", index=False, encoding="utf-8-sig")
                # Save immediately so Ctrl+C during later calibration does not lose the model.
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "architecture": "talc_unet_resnet18",
                        "threshold": 0.5,
                        "crop_size": args.crop_size,
                        "calibrated": False,
                        "validation_summary_at_save": validation_summary,
                    },
                    args.model_path,
                )
            else:
                epochs_without_improvement += 1

            pd.DataFrame(history).to_csv(
                args.output_dir / "training_history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if epochs_without_improvement >= 4:
                print("Early stopping.")
                break
        if epochs_without_improvement >= 4:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    # Run expensive neural-network inference only once per validation image.
    validation_cache = build_prediction_cache(
        model,
        validation_frame,
        device,
        tile_size=args.crop_size,
        overlap=min(args.overlap, args.crop_size // 3),
        description="cache validation probabilities",
    )

    # Calibrate the pixel-probability threshold only on validation.
    threshold_rows = []
    for threshold in np.arange(0.25, 0.76, 0.05):
        _, summary = evaluate_cached(validation_cache, float(threshold))
        threshold_rows.append({"threshold": float(threshold), **summary})
    threshold_frame = pd.DataFrame(threshold_rows).sort_values(
        ["selection_score", "positive_mean_iou", "threshold_f1"],
        ascending=False,
    )
    best_threshold = float(threshold_frame.iloc[0]["threshold"])
    validation_metrics, validation_summary = evaluate_cached(
        validation_cache, best_threshold
    )

    # Test remains untouched until the threshold has been selected.
    test_cache = build_prediction_cache(
        model,
        test_frame,
        device,
        tile_size=args.crop_size,
        overlap=min(args.overlap, args.crop_size // 3),
        description="cache test probabilities",
    )
    test_metrics, test_summary = evaluate_cached(test_cache, best_threshold)

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": "talc_unet_resnet18",
        "threshold": best_threshold,
        "crop_size": args.crop_size,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "calibrated": True,
    }
    torch.save(checkpoint, args.model_path)

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    threshold_frame.to_csv(args.output_dir / "threshold_search.csv", index=False, encoding="utf-8-sig")
    validation_metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(args.output_dir / "test_metrics.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_threshold": best_threshold,
                "validation": validation_summary,
                "test": test_summary,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("TRAINING FINISHED")
    print("Best threshold:", best_threshold)
    print("Test positive mean IoU:", f"{test_summary['positive_mean_iou']:.4f}")
    print("Test positive mean Dice:", f"{test_summary['positive_mean_dice']:.4f}")
    print("Test positive HD95:", f"{test_summary['positive_mean_hausdorff95_pixels']:.2f}px")
    print("Test positive area MAE:", f"{test_summary['positive_area_mae_pp']:.2f} p.p.")
    print("Test negative predicted area:", f"{test_summary['negative_mean_predicted_share_percent']:.2f}%")
    print("Test >10% F1:", f"{test_summary['threshold_f1']:.4f}")
    print("Test >10% precision/recall:", f"{test_summary['threshold_precision']:.4f}/{test_summary['threshold_recall']:.4f}")
    print("Model:", args.model_path.resolve())


if __name__ == "__main__":
    main()
