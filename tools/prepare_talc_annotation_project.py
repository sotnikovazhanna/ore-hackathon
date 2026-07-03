from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е").replace("_", " ").replace("-", " ").strip()


def source_part(path: Path) -> str:
    joined = " / ".join(normalize_text(part) for part in path.parts)
    if "ч1" in joined or "часть 1" in joined or "part 1" in joined:
        return "part_1"
    if "ч2" in joined or "часть 2" in joined or "part 2" in joined:
        return "part_2"
    return "unknown"


def is_talc_annotation(path: Path) -> bool:
    joined = " / ".join(normalize_text(part) for part in path.parts)
    return "отальк" in joined and "област" in joined


def is_talc_original(path: Path) -> bool:
    joined = " / ".join(normalize_text(part) for part in path.parts)
    return "отальк" in joined and "област" not in joined


def task_id(relative_path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", relative_path.stem).strip("_")[:40]
    digest = hashlib.md5(str(relative_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create portable talc annotation tasks.")
    parser.add_argument("--data-root", required=True, type=Path, help="Root folder of the downloaded hackathon dataset")
    parser.add_argument("--output", type=Path, default=Path("data/manual_talc/tasks.csv"))
    parser.add_argument(
        "--assignees",
        default="",
        help="Comma-separated names for round-robin assignment, e.g. Katya,Zhanna,Maria,Anna",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    images = [
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    originals = [path for path in images if is_talc_original(path)]
    annotations = [path for path in images if is_talc_annotation(path)]

    originals_by_key: dict[tuple[str, str], list[Path]] = {}
    for path in originals:
        key = (source_part(path), path.name.lower())
        originals_by_key.setdefault(key, []).append(path)

    rows: list[dict[str, str]] = []
    assignees = [name.strip() for name in args.assignees.split(",") if name.strip()]

    for index, annotated in enumerate(sorted(annotations)):
        key = (source_part(annotated), annotated.name.lower())
        candidates = originals_by_key.get(key, [])
        if not candidates:
            # Fallback by filename if part naming differs.
            candidates = [path for path in originals if path.name.lower() == annotated.name.lower()]
        if not candidates:
            print(f"WARNING: original not found for {annotated}")
            continue

        original = candidates[0]
        original_rel = original.relative_to(data_root)
        annotated_rel = annotated.relative_to(data_root)

        rows.append(
            {
                "task_id": task_id(original_rel),
                "filename": original.name,
                "source_part": source_part(original),
                "original_relative": str(original_rel),
                "annotated_relative": str(annotated_rel),
                "assignee": assignees[index % len(assignees)] if assignees else "",
                "status": "pending",
            }
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["source_part", "filename"]).reset_index(drop=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Created {len(frame)} annotation tasks: {output.resolve()}")
    if assignees:
        print(frame["assignee"].value_counts().to_string())


if __name__ == "__main__":
    main()
