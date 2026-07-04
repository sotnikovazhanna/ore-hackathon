from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from src.talc_unet import TalcUNetResNet18

ProgressCallback = Callable[[float, str], None]
PIPELINE_API_VERSION = "orescope-final-v1"

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)


@dataclass
class AnalysisResult:
    ore_class: str
    decision_reason: str
    warning: str
    quality_label: str
    quality_score: float
    quality_reason: str
    talc_share_raw_percent: float
    talc_share_percent: float
    ordinary_share_percent: float
    fine_share_percent: float
    sulfide_share_percent: float
    ordinary_fraction_of_sulfides: float
    fine_fraction_of_sulfides: float
    talc_uncertain_fraction_percent: float
    ordinary_fine_margin: float
    overlay: np.ndarray
    confidence_map: np.ndarray
    segmentation_model: str
    segmentation_is_manual: bool
    tile_count: int


class OreAnalyzer:
    API_VERSION = PIPELINE_API_VERSION

    def __init__(
        self,
        model_dir: str | Path = "models",
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.classifier_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.intergrowth_model, _ = self._load_binary_classifier(
            self.model_dir / "ordinary_fine_mobilenet_v3_small.pth"
        )

        (
            self.talc_segmenter,
            self.talc_segmentation_kind,
            self.talc_threshold,
            self.segmentation_is_manual,
        ) = self._load_talc_segmenter()

        self.area_calibration = self._load_area_calibration()

    def model_status(self) -> dict[str, object]:
        required_ready = (
            self.intergrowth_model is not None
            and self.talc_segmenter is not None
        )
        return {
            "api_version": self.API_VERSION,
            "device": str(self.device),
            "ready": required_ready,
            "intergrowth_loaded": self.intergrowth_model is not None,
            "segmentation_loaded": self.talc_segmenter is not None,
            "segmentation_model": self.talc_segmentation_kind,
            "segmentation_is_manual": self.segmentation_is_manual,
            "default_talc_mask_threshold": self.talc_threshold,
            "area_calibration": self.area_calibration,
        }

    def _load_area_calibration(self) -> dict[str, float] | None:
        path = self.model_dir / "talc_area_calibration.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "slope": float(payload.get("slope", 1.0)),
                "intercept": float(payload.get("intercept", 0.0)),
                "minimum": float(payload.get("minimum", 0.0)),
                "maximum": float(payload.get("maximum", 100.0)),
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _calibrate_area(self, raw_share: float) -> float:
        if self.area_calibration is None:
            return float(raw_share)
        value = (
            self.area_calibration["slope"] * raw_share
            + self.area_calibration["intercept"]
        )
        return float(
            np.clip(
                value,
                self.area_calibration["minimum"],
                self.area_calibration["maximum"],
            )
        )

    def _load_binary_classifier(self, path: Path):
        if not path.exists():
            return None, None
        checkpoint = torch.load(path, map_location=self.device)
        model = models.mobilenet_v3_small(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(input_features, 2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()
        threshold = checkpoint.get("decision_threshold")
        return model, float(threshold) if threshold is not None else None

    def _load_talc_segmenter(self):
        manual_path = self.model_dir / "talc_unet_resnet18.pth"
        if manual_path.exists():
            checkpoint = torch.load(manual_path, map_location=self.device)
            model = TalcUNetResNet18(pretrained=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()
            return (
                model,
                "manual_unet_resnet18",
                float(checkpoint.get("threshold", 0.70)),
                True,
            )

        legacy_path = self.model_dir / "talc_lraspp_mobilenet_v3.pth"
        if legacy_path.exists():
            checkpoint = torch.load(legacy_path, map_location=self.device)
            model = lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=2,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device).eval()
            return model, "legacy_lraspp", 0.55, False

        return None, "none", 0.50, False

    @staticmethod
    def _positions(length: int, tile_size: int, stride: int) -> list[int]:
        if length <= tile_size:
            return [0]
        result = list(range(0, length - tile_size + 1, stride))
        if result[-1] != length - tile_size:
            result.append(length - tile_size)
        return result

    @staticmethod
    def _extract_tile(
        image: np.ndarray,
        y: int,
        x: int,
        tile_size: int,
    ) -> tuple[np.ndarray, int, int]:
        tile = image[y : y + tile_size, x : x + tile_size]
        valid_h, valid_w = tile.shape[:2]
        if valid_h == tile_size and valid_w == tile_size:
            return tile, valid_h, valid_w
        tile = cv2.copyMakeBorder(
            tile,
            0,
            tile_size - valid_h,
            0,
            tile_size - valid_w,
            cv2.BORDER_REFLECT_101,
        )
        return tile, valid_h, valid_w

    def _classifier_probability(self, model, rgb: np.ndarray) -> float:
        if model is None:
            return 0.5
        tensor = (
            self.classifier_transform(Image.fromarray(rgb))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            probability = torch.softmax(model(tensor), dim=1)[0, 1]
        return float(probability.cpu())

    def _tiles_to_tensor(self, tiles: list[np.ndarray]) -> torch.Tensor:
        array = np.stack(tiles)
        tensor = (
            torch.from_numpy(array)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def _classifier_batch_probability(
        self,
        model,
        tiles: list[np.ndarray],
    ) -> np.ndarray:
        if model is None:
            return np.full(len(tiles), 0.5, dtype=np.float32)
        tensor = (
            torch.from_numpy(np.stack(tiles))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        tensor = F.interpolate(
            tensor,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor[:, :, 16:240, 16:240]
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            probability = torch.softmax(model(tensor), dim=1)[:, 1]
        return probability.cpu().numpy().astype(np.float32)

    def _segment_talc_batch(self, tiles: list[np.ndarray]) -> np.ndarray:
        if self.talc_segmenter is None:
            h, w = tiles[0].shape[:2]
            return np.zeros((len(tiles), h, w), dtype=np.float32)

        tensor = self._tiles_to_tensor(tiles).to(self.device)
        with torch.inference_mode():
            if self.talc_segmentation_kind == "manual_unet_resnet18":
                probability = torch.sigmoid(self.talc_segmenter(tensor))[:, 0]
            else:
                logits = self.talc_segmenter(tensor)["out"]
                probability = torch.softmax(logits, dim=1)[:, 1]
        return probability.cpu().numpy().astype(np.float32)

    @staticmethod
    def _sulfide_mask(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness = lab[..., 0]
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(lightness)
        otsu_value, _ = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        percentile_value = np.percentile(enhanced, 82)
        threshold = max(float(otsu_value), float(percentile_value))
        mask = (enhanced >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        return mask

    @staticmethod
    def _postprocess_talc(mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((7, 7), np.uint8),
            iterations=1,
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        result = np.zeros_like(mask)
        min_area = max(25, int(mask.size * 0.00035))
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= min_area:
                result[labels == component] = 1
        return result.astype(bool)

    @staticmethod
    def _quality_assessment(
        talc_share: float,
        talc_uncertain_fraction: float,
        sulfide_share: float,
        ordinary_fine_margin: float,
        segmentation_available: bool,
        intergrowth_available: bool,
    ) -> tuple[str, float, str, str]:
        score = 100.0
        reasons: list[str] = []
        warnings: list[str] = []

        if not segmentation_available:
            score -= 65
            reasons.append("модель оценки зоны оталькования недоступна")
        if not intergrowth_available:
            score -= 50
            reasons.append("модель типов срастаний недоступна")

        distance_to_rule = abs(talc_share - 10.0)
        if distance_to_rule <= 2.0:
            score -= 35
            reasons.append("доля талька близка к порогу 10%")
            warnings.append(
                "Доля талька близка к порогу 10%; рекомендуется визуальная проверка маски."
            )
        elif distance_to_rule <= 4.0:
            score -= 15
            reasons.append("доля талька находится недалеко от порога 10%")

        if talc_uncertain_fraction >= 15.0:
            score -= 25
            reasons.append("много участков с неуверенным прогнозом талька")
        elif talc_uncertain_fraction >= 7.0:
            score -= 10
            reasons.append("есть участки с неуверенным прогнозом талька")

        if sulfide_share < 0.2:
            score -= 20
            reasons.append("обнаружено очень мало сульфидной фазы")
            warnings.append(
                "Сульфидной фазы обнаружено мало; вывод о типе срастаний менее устойчив."
            )

        if ordinary_fine_margin < 0.12:
            score -= 25
            reasons.append("обычные и тонкие срастания различаются слабо")
            warnings.append(
                "Доли обычных и тонких срастаний близки; рекомендуется экспертная проверка."
            )
        elif ordinary_fine_margin < 0.25:
            score -= 10
            reasons.append("разделение обычных и тонких срастаний умеренное")

        score = float(np.clip(score, 0.0, 100.0))
        if score >= 75:
            label = "Высокая"
        elif score >= 50:
            label = "Средняя"
        else:
            label = "Требуется проверка"

        if not reasons:
            reasons.append("оценка находится вдали от пороговых значений")

        return (
            label,
            score,
            "; ".join(reasons),
            " ".join(dict.fromkeys(warnings)),
        )

    def analyze(
        self,
        image: np.ndarray,
        tile_size: int = 384,
        overlap: int = 32,
        batch_size: int | None = None,
        talc_mask_threshold: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape H×W×3.")

        h, w = image.shape[:2]
        if tile_size <= overlap:
            raise ValueError("tile_size must be greater than overlap.")

        threshold_mask = (
            self.talc_threshold
            if talc_mask_threshold is None
            else float(talc_mask_threshold)
        )
        if batch_size is None:
            batch_size = 12 if self.device.type == "cuda" else 2

        whole_ordinary_probability = self._classifier_probability(
            self.intergrowth_model,
            image,
        )

        stride = tile_size - overlap
        coordinates = [
            (y, x)
            for y in self._positions(h, tile_size, stride)
            for x in self._positions(w, tile_size, stride)
        ]

        ordinary_accumulator = np.zeros((h, w), dtype=np.float32)
        talc_accumulator = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.uint16)

        total_batches = max(
            1,
            (len(coordinates) + batch_size - 1) // batch_size,
        )

        for batch_index, start in enumerate(
            range(0, len(coordinates), batch_size)
        ):
            batch_coordinates = coordinates[start : start + batch_size]
            tiles: list[np.ndarray] = []
            valid_sizes: list[tuple[int, int]] = []
            for y, x in batch_coordinates:
                tile, valid_h, valid_w = self._extract_tile(
                    image,
                    y,
                    x,
                    tile_size,
                )
                tiles.append(tile)
                valid_sizes.append((valid_h, valid_w))

            ordinary_probabilities = self._classifier_batch_probability(
                self.intergrowth_model,
                tiles,
            )
            talc_probabilities = self._segment_talc_batch(tiles)

            for index, ((y, x), (valid_h, valid_w)) in enumerate(
                zip(batch_coordinates, valid_sizes)
            ):
                ordinary_probability = (
                    0.75 * float(ordinary_probabilities[index])
                    + 0.25 * whole_ordinary_probability
                )
                ordinary_accumulator[
                    y : y + valid_h,
                    x : x + valid_w,
                ] += ordinary_probability
                talc_accumulator[
                    y : y + valid_h,
                    x : x + valid_w,
                ] += talc_probabilities[index, :valid_h, :valid_w]
                weights[y : y + valid_h, x : x + valid_w] += 1

            if progress_callback is not None:
                progress_callback(
                    (batch_index + 1) / total_batches,
                    "Анализ участков изображения",
                )

        safe_weights = np.maximum(weights.astype(np.float32), 1.0)
        ordinary_map = ordinary_accumulator / safe_weights
        fine_map = 1.0 - ordinary_map
        talc_map = talc_accumulator / safe_weights

        sulfide_mask = self._sulfide_mask(image)
        ordinary_mask = (ordinary_map >= 0.5) & (sulfide_mask > 0)
        fine_mask = (ordinary_map < 0.5) & (sulfide_mask > 0)

        talc_mask = talc_map >= threshold_mask
        talc_mask &= sulfide_mask == 0
        talc_mask = self._postprocess_talc(talc_mask)

        total_pixels = float(h * w)
        ordinary_share = float(ordinary_mask.sum() / total_pixels * 100.0)
        fine_share = float(fine_mask.sum() / total_pixels * 100.0)
        sulfide_share = float(
            (ordinary_mask | fine_mask).sum() / total_pixels * 100.0
        )
        talc_share_raw = float(talc_mask.sum() / total_pixels * 100.0)
        talc_share = self._calibrate_area(talc_share_raw)

        if sulfide_share > 0:
            ordinary_fraction = ordinary_share / sulfide_share * 100.0
            fine_fraction = fine_share / sulfide_share * 100.0
            sulfide_pixels = sulfide_mask.astype(bool)
            ordinary_fine_margin = float(
                np.mean(np.abs(ordinary_map[sulfide_pixels] - 0.5) * 2.0)
            )
        else:
            ordinary_fraction = 0.0
            fine_fraction = 0.0
            ordinary_fine_margin = float(
                abs(whole_ordinary_probability - 0.5) * 2.0
            )

        non_sulfide = sulfide_mask == 0
        if np.any(non_sulfide):
            talc_uncertain_fraction = float(
                np.mean(
                    np.abs(talc_map[non_sulfide] - threshold_mask) <= 0.08
                )
                * 100.0
            )
        else:
            talc_uncertain_fraction = 0.0

        if talc_share > 10.0:
            ore_class = "Оталькованная руда"
            decision_reason = (
                f"доля зоны оталькования {talc_share:.1f}% превышает "
                "экспертный порог 10%"
            )
        else:
            if sulfide_share > 0:
                ordinary_dominates = ordinary_share >= fine_share
            else:
                ordinary_dominates = whole_ordinary_probability >= 0.5

            if ordinary_dominates:
                ore_class = "Рядовая руда"
                decision_reason = (
                    f"доля зоны оталькования {talc_share:.1f}% не превышает 10%; "
                    f"среди обнаруженной сульфидной фазы преобладают обычные "
                    f"срастания ({ordinary_fraction:.1f}% против "
                    f"{fine_fraction:.1f}%)"
                )
            else:
                ore_class = "Труднообогатимая руда"
                decision_reason = (
                    f"доля зоны оталькования {talc_share:.1f}% не превышает 10%; "
                    f"среди обнаруженной сульфидной фазы преобладают тонкие "
                    f"срастания ({fine_fraction:.1f}% против "
                    f"{ordinary_fraction:.1f}%)"
                )

        (
            quality_label,
            quality_score,
            quality_reason,
            warning,
        ) = self._quality_assessment(
            talc_share=talc_share,
            talc_uncertain_fraction=talc_uncertain_fraction,
            sulfide_share=sulfide_share,
            ordinary_fine_margin=ordinary_fine_margin,
            segmentation_available=self.talc_segmenter is not None,
            intergrowth_available=self.intergrowth_model is not None,
        )

        overlay = image.copy().astype(np.float32)
        colors = {
            "ordinary": np.array([0, 255, 0], dtype=np.float32),
            "fine": np.array([255, 65, 30], dtype=np.float32),
            "talc": np.array([0, 80, 255], dtype=np.float32),
        }
        for mask, color in [
            (ordinary_mask, colors["ordinary"]),
            (fine_mask, colors["fine"]),
            (talc_mask, colors["talc"]),
        ]:
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        talc_confidence = np.clip(
            np.abs(talc_map - threshold_mask)
            / max(threshold_mask, 1.0 - threshold_mask),
            0.0,
            1.0,
        )
        ordinary_confidence = np.clip(
            np.abs(ordinary_map - 0.5) * 2.0,
            0.0,
            1.0,
        )
        confidence_map = talc_confidence.astype(np.float32)
        confidence_map[sulfide_mask > 0] = ordinary_confidence[
            sulfide_mask > 0
        ]

        return AnalysisResult(
            ore_class=ore_class,
            decision_reason=decision_reason,
            warning=warning,
            quality_label=quality_label,
            quality_score=quality_score,
            quality_reason=quality_reason,
            talc_share_raw_percent=float(talc_share_raw),
            talc_share_percent=float(talc_share),
            ordinary_share_percent=float(ordinary_share),
            fine_share_percent=float(fine_share),
            sulfide_share_percent=float(sulfide_share),
            ordinary_fraction_of_sulfides=float(ordinary_fraction),
            fine_fraction_of_sulfides=float(fine_fraction),
            talc_uncertain_fraction_percent=float(talc_uncertain_fraction),
            ordinary_fine_margin=float(ordinary_fine_margin),
            overlay=overlay,
            confidence_map=confidence_map,
            segmentation_model=self.talc_segmentation_kind,
            segmentation_is_manual=self.segmentation_is_manual,
            tile_count=len(coordinates),
        )
