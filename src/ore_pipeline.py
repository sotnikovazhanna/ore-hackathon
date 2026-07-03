from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


@dataclass
class AnalysisResult:
    ore_class: str
    talc_probability: float
    talc_share_percent: float
    ordinary_share_percent: float
    fine_share_percent: float
    sulfide_share_percent: float
    overlay: np.ndarray
    confidence_map: np.ndarray


class OreAnalyzer:
    def __init__(
        self,
        model_dir: str | Path = "models",
        device: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.classifier_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.intergrowth_model = self._load_binary_classifier(
            self.model_dir
            / "ordinary_fine_mobilenet_v3_small.pth"
        )

        self.talc_gate_model = self._load_binary_classifier(
            self.model_dir
            / "talc_gate_mobilenet_v3_small.pth"
        )

        self.talc_segmenter, self.talc_threshold = (
            self._load_talc_segmenter()
        )

    def _load_binary_classifier(self, path: Path):
        if not path.exists():
            return None

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        model = models.mobilenet_v3_small(
            weights=None,
        )
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(
            input_features,
            2,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(self.device).eval()
        return model

    def _load_talc_segmenter(self):
        path = (
            self.model_dir
            / "talc_lraspp_mobilenet_v3.pth"
        )

        if not path.exists():
            return None, 0.5

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(self.device).eval()

        # Для демонстрации используем более консервативный порог,
        # чтобы не заливать всю тёмную матрицу.
        return model, 0.55

    @staticmethod
    def _positions(length: int, tile_size: int, stride: int):
        if length <= tile_size:
            return [0]

        positions = list(
            range(0, length - tile_size + 1, stride)
        )

        if positions[-1] != length - tile_size:
            positions.append(length - tile_size)

        return positions

    @staticmethod
    def _extract_tile(
        image: np.ndarray,
        y: int,
        x: int,
        tile_size: int,
    ):
        tile = image[
            y:y + tile_size,
            x:x + tile_size,
        ]

        h, w = tile.shape[:2]

        if h == tile_size and w == tile_size:
            return tile, h, w

        padded = cv2.copyMakeBorder(
            tile,
            0,
            tile_size - h,
            0,
            tile_size - w,
            cv2.BORDER_REFLECT_101,
        )
        return padded, h, w

    def _classifier_probability(
        self,
        model,
        tile: np.ndarray,
    ) -> float:
        if model is None:
            return 0.0

        tensor = self.classifier_transform(
            Image.fromarray(tile)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probability = torch.softmax(
                logits,
                dim=1,
            )[0, 1]

        return float(probability.cpu())

    def _segment_talc_tile(
        self,
        tile: np.ndarray,
    ) -> np.ndarray:
        if self.talc_segmenter is None:
            return np.zeros(
                tile.shape[:2],
                dtype=np.float32,
            )

        tensor = torch.from_numpy(
            tile.transpose(2, 0, 1).copy()
        ).float() / 255.0

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        tensor = (
            (tensor - mean) / std
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.talc_segmenter(
                tensor
            )["out"]

            probability = torch.softmax(
                logits,
                dim=1,
            )[0, 1]

        return probability.cpu().numpy()

    @staticmethod
    def _sulfide_mask(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2LAB,
        )
        lightness = lab[..., 0]

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        enhanced = clahe.apply(lightness)

        otsu_value, otsu_mask = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        percentile_value = np.percentile(
            enhanced,
            82,
        )

        threshold = max(
            float(otsu_value),
            float(percentile_value),
        )

        mask = (
            enhanced >= threshold
        ).astype(np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )

        return mask

    def analyze(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 96,
        talc_gate_threshold: float = 0.97,
        talc_mask_threshold: float | None = None,
    ) -> AnalysisResult:
        image = np.asarray(image, dtype=np.uint8)
        h, w = image.shape[:2]

        stride = tile_size - overlap
        ys = self._positions(h, tile_size, stride)
        xs = self._positions(w, tile_size, stride)

        ordinary_accumulator = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        fine_accumulator = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        talc_accumulator = np.zeros(
            (h, w),
            dtype=np.float32,
        )
        weights = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        tile_talc_probabilities = []

        for y in ys:
            for x in xs:
                tile, valid_h, valid_w = self._extract_tile(
                    image,
                    y,
                    x,
                    tile_size,
                )

                ordinary_probability = (
                    self._classifier_probability(
                        self.intergrowth_model,
                        tile,
                    )
                )
                fine_probability = (
                    1.0 - ordinary_probability
                )

                gate_probability = (
                    self._classifier_probability(
                        self.talc_gate_model,
                        tile,
                    )
                )
                tile_talc_probabilities.append(
                    gate_probability
                )

                talc_probability = (
                    self._segment_talc_tile(tile)
                )[:valid_h, :valid_w]

                ordinary_accumulator[
                    y:y + valid_h,
                    x:x + valid_w,
                ] += ordinary_probability

                fine_accumulator[
                    y:y + valid_h,
                    x:x + valid_w,
                ] += fine_probability

                talc_accumulator[
                    y:y + valid_h,
                    x:x + valid_w,
                ] += talc_probability

                weights[
                    y:y + valid_h,
                    x:x + valid_w,
                ] += 1

        weights = np.maximum(weights, 1)

        ordinary_map = (
            ordinary_accumulator / weights
        )
        fine_map = (
            fine_accumulator / weights
        )
        talc_map = (
            talc_accumulator / weights
        )

        sulfide_mask = self._sulfide_mask(image)

        ordinary_mask = (
            (ordinary_map >= fine_map)
            & (sulfide_mask > 0)
        )
        fine_mask = (
            (fine_map > ordinary_map)
            & (sulfide_mask > 0)
        )

        threshold = (
            self.talc_threshold
            if talc_mask_threshold is None
            else talc_mask_threshold
        )

        talc_mask = talc_map >= threshold

        # Тальк не должен перекрывать найденную светлую рудную фазу.
        talc_mask = talc_mask & (sulfide_mask == 0)

        talc_mask = cv2.morphologyEx(
            talc_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        ).astype(bool)

        total_pixels = h * w

        ordinary_share = (
            ordinary_mask.sum()
            / total_pixels
            * 100
        )
        fine_share = (
            fine_mask.sum()
            / total_pixels
            * 100
        )
        sulfide_share = (
            (ordinary_mask | fine_mask).sum()
            / total_pixels
            * 100
        )
        talc_share = (
            talc_mask.sum()
            / total_pixels
            * 100
        )

        # Верхний квартиль tile-level вероятностей лучше отражает
        # локальные зоны, чем простое среднее по большой панораме.
        talc_probability = float(
            np.quantile(
                tile_talc_probabilities,
                0.75,
            )
            if tile_talc_probabilities
            else 0.0
        )

        if (
            talc_probability
            >= talc_gate_threshold
        ):
            ore_class = "Оталькованная руда"
        elif ordinary_share >= fine_share:
            ore_class = "Рядовая руда"
        else:
            ore_class = "Труднообогатимая руда"

        overlay = image.copy().astype(np.float32)

        colors = {
            "ordinary": np.array(
                [0, 255, 0],
                dtype=np.float32,
            ),
            "fine": np.array(
                [255, 0, 0],
                dtype=np.float32,
            ),
            "talc": np.array(
                [0, 80, 255],
                dtype=np.float32,
            ),
        }

        alpha = 0.45

        for mask, color in [
            (ordinary_mask, colors["ordinary"]),
            (fine_mask, colors["fine"]),
            (talc_mask, colors["talc"]),
        ]:
            overlay[mask] = (
                (1 - alpha) * overlay[mask]
                + alpha * color
            )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        confidence_map = np.maximum.reduce([
            ordinary_map,
            fine_map,
            talc_map,
        ])

        return AnalysisResult(
            ore_class=ore_class,
            talc_probability=talc_probability,
            talc_share_percent=float(talc_share),
            ordinary_share_percent=float(ordinary_share),
            fine_share_percent=float(fine_share),
            sulfide_share_percent=float(sulfide_share),
            overlay=overlay,
            confidence_map=confidence_map,
        )
