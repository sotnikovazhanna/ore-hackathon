from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

WINDOW_NAME = "Talc annotation"
HEADER_HEIGHT = 72
MAX_PANE_WIDTH = 760
MAX_IMAGE_HEIGHT = 820


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", text.strip())
    return cleaned or "annotator"


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = rgb.astype(np.float32).copy()
    selected = mask.astype(bool)
    overlay[selected] = 0.55 * overlay[selected] + 0.45 * np.array([0, 90, 255], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path, quality=92)


def fit_image(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(MAX_PANE_WIDTH / w, MAX_IMAGE_HEIGHT / h, 1.0)
    resized = cv2.resize(
        rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    return resized, scale


def draw_mask_overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 90, 255), alpha=0.40) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = (1 - alpha) * result[selected] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


class AnnotationSession:
    def __init__(
        self,
        data_root: Path,
        tasks_path: Path,
        output_root: Path,
        assignee: str,
        pseudo_mask_root: Path | None,
    ) -> None:
        self.data_root = data_root.resolve()
        self.tasks_path = tasks_path
        self.output_root = output_root
        self.assignee = assignee
        self.assignee_key = safe_name(assignee)
        self.pseudo_mask_root = pseudo_mask_root

        tasks = pd.read_csv(tasks_path, encoding="utf-8-sig")
        if assignee:
            tasks = tasks[(tasks["assignee"].fillna("") == assignee) | (tasks["assignee"].fillna("") == "")]
        self.tasks = tasks.reset_index(drop=True)
        if self.tasks.empty:
            raise RuntimeError(f"No tasks found for assignee '{assignee}'.")

        self.status_path = output_root / "status" / f"status_{self.assignee_key}.csv"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        if self.status_path.exists():
            saved_status = pd.read_csv(self.status_path, encoding="utf-8-sig")
            status_map = saved_status.set_index("task_id").to_dict("index")
            for index, row in self.tasks.iterrows():
                old = status_map.get(row["task_id"])
                if old:
                    for column in ["status", "mask_path", "mask_share_percent", "updated_at"]:
                        if column in old:
                            self.tasks.loc[index, column] = old[column]

        for column, default in [
            ("status", "pending"),
            ("mask_path", ""),
            ("mask_share_percent", np.nan),
            ("updated_at", ""),
        ]:
            if column not in self.tasks:
                self.tasks[column] = default

        pending = self.tasks.index[self.tasks["status"].fillna("pending") != "saved"].tolist()
        self.index = pending[0] if pending else 0

        self.rgb: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.auto_mask: np.ndarray | None = None
        self.points: list[tuple[int, int]] = []
        self.scale = 1.0
        self.left_width = 0
        self.left_height = 0
        self.show_auto = False

    def mask_path(self, task_id: str) -> Path:
        return self.output_root / "annotations" / self.assignee_key / "masks" / f"{task_id}.png"

    def overlay_path(self, task_id: str) -> Path:
        return self.output_root / "annotations" / self.assignee_key / "overlays" / f"{task_id}.jpg"

    def current_row(self) -> pd.Series:
        return self.tasks.iloc[self.index]

    def load_current(self) -> None:
        row = self.current_row()
        original_path = self.data_root / Path(row["original_relative"])
        reference_path = self.data_root / Path(row["annotated_relative"])

        if not original_path.exists():
            raise FileNotFoundError(f"Original not found: {original_path}")
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference not found: {reference_path}")

        self.rgb = load_rgb(original_path)
        self.reference = load_rgb(reference_path)
        self.mask = load_mask(self.mask_path(row["task_id"]), self.rgb.shape[:2])
        self.points = []

        self.auto_mask = np.zeros(self.rgb.shape[:2], dtype=np.uint8)
        if self.pseudo_mask_root:
            candidates = [
                self.pseudo_mask_root / f"{Path(row['filename']).stem}_mask.png",
                self.pseudo_mask_root / f"{row['task_id']}.png",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.auto_mask = load_mask(candidate, self.rgb.shape[:2])
                    break

    def save_status(self) -> None:
        self.tasks.to_csv(self.status_path, index=False, encoding="utf-8-sig")

    def save_current(self) -> None:
        assert self.rgb is not None and self.mask is not None
        row = self.current_row()
        mask_path = self.mask_path(row["task_id"])
        overlay_path = self.overlay_path(row["task_id"])
        save_mask(self.mask, mask_path)
        save_overlay(self.rgb, self.mask, overlay_path)

        self.tasks.loc[self.index, "status"] = "saved"
        self.tasks.loc[self.index, "mask_path"] = str(mask_path)
        self.tasks.loc[self.index, "mask_share_percent"] = float(self.mask.mean() * 100)
        self.tasks.loc[self.index, "updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.save_status()

    def set_skipped(self) -> None:
        self.tasks.loc[self.index, "status"] = "skipped"
        self.tasks.loc[self.index, "updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self.save_status()

    def move(self, step: int) -> None:
        self.index = int(np.clip(self.index + step, 0, len(self.tasks) - 1))
        self.load_current()

    def commit_polygon(self) -> None:
        assert self.mask is not None
        if len(self.points) < 3:
            return
        polygon = np.asarray(self.points, dtype=np.int32)
        cv2.fillPoly(self.mask, [polygon], 1)
        self.points = []

    def render(self) -> np.ndarray:
        assert self.rgb is not None and self.reference is not None and self.mask is not None

        left = draw_mask_overlay(self.rgb, self.mask)
        if self.show_auto and self.auto_mask is not None:
            left = draw_mask_overlay(left, self.auto_mask, color=(255, 160, 0), alpha=0.25)

        left_resized, self.scale = fit_image(left)
        reference_resized = cv2.resize(
            self.reference,
            (left_resized.shape[1], left_resized.shape[0]),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR,
        )
        self.left_height, self.left_width = left_resized.shape[:2]

        # Draw the currently open polygon on the left pane.
        if self.points:
            scaled_points = np.asarray(
                [(int(x * self.scale), int(y * self.scale)) for x, y in self.points],
                dtype=np.int32,
            )
            for point in scaled_points:
                cv2.circle(left_resized, tuple(point), 4, (255, 255, 0), -1)
            if len(scaled_points) >= 2:
                cv2.polylines(left_resized, [scaled_points], False, (255, 255, 0), 2)

        gap = 12
        canvas_width = self.left_width * 2 + gap
        canvas = np.full((HEADER_HEIGHT + self.left_height, canvas_width, 3), 245, dtype=np.uint8)
        canvas[HEADER_HEIGHT:, : self.left_width] = cv2.cvtColor(left_resized, cv2.COLOR_RGB2BGR)
        canvas[HEADER_HEIGHT:, self.left_width + gap :] = cv2.cvtColor(reference_resized, cv2.COLOR_RGB2BGR)

        row = self.current_row()
        title = f"{self.index + 1}/{len(self.tasks)}  {row['filename']}  assignee={self.assignee or 'all'}  status={row.get('status', 'pending')}"
        controls = "Left click: vertex | Enter: close polygon | Right click/U: undo | C: clear | K: use auto mask | A: toggle auto | S: save+next | N: skip | B: previous | Q: quit"
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, controls, (10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, "DRAW HERE", (10, HEADER_HEIGHT + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "BLUE-LINE REFERENCE", (self.left_width + gap + 10, HEADER_HEIGHT + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if y < HEADER_HEIGHT or x >= self.left_width or x < 0:
            return
        original_x = int(round(x / self.scale))
        original_y = int(round((y - HEADER_HEIGHT) / self.scale))
        assert self.rgb is not None
        original_x = int(np.clip(original_x, 0, self.rgb.shape[1] - 1))
        original_y = int(np.clip(original_y, 0, self.rgb.shape[0] - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((original_x, original_y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def run(self) -> None:
        self.load_current()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKeyEx(30)
            if key < 0:
                continue
            key_low = key & 0xFF

            if key_low in (13, 10):  # Enter
                self.commit_polygon()
            elif key_low in (ord("u"), ord("U"), 8):
                if self.points:
                    self.points.pop()
            elif key_low in (ord("c"), ord("C")):
                assert self.mask is not None
                self.mask[:] = 0
                self.points = []
            elif key_low in (ord("k"), ord("K")):
                if self.auto_mask is not None and self.auto_mask.any():
                    self.mask = self.auto_mask.copy()
                    self.points = []
            elif key_low in (ord("a"), ord("A")):
                self.show_auto = not self.show_auto
            elif key_low in (ord("s"), ord("S")):
                self.commit_polygon()
                self.save_current()
                if self.index >= len(self.tasks) - 1:
                    print("All assigned tasks are complete.")
                    break
                self.move(1)
            elif key_low in (ord("n"), ord("N")):
                self.set_skipped()
                if self.index >= len(self.tasks) - 1:
                    break
                self.move(1)
            elif key_low in (ord("b"), ord("B")):
                self.move(-1)
            elif key_low in (ord("q"), ord("Q"), 27):
                self.save_status()
                break

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual polygon annotation for talc regions.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("data/manual_talc/tasks.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/manual_talc"))
    parser.add_argument("--assignee", default="")
    parser.add_argument("--pseudo-mask-root", type=Path, default=Path("data/processed/talc_masks"))
    args = parser.parse_args()

    session = AnnotationSession(
        data_root=args.data_root,
        tasks_path=args.tasks,
        output_root=args.output_root,
        assignee=args.assignee,
        pseudo_mask_root=args.pseudo_mask_root if args.pseudo_mask_root.exists() else None,
    )
    session.run()


if __name__ == "__main__":
    main()
