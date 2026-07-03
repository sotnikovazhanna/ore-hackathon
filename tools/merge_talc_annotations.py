from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-assignee talc annotation status files.")
    parser.add_argument("--root", type=Path, default=Path("data/manual_talc"))
    parser.add_argument("--output", type=Path, default=Path("data/manual_talc/merged_annotations.csv"))
    args = parser.parse_args()

    status_dir = args.root / "status"
    files = sorted(status_dir.glob("status_*.csv"))
    if not files:
        raise FileNotFoundError(f"No status files found in {status_dir}")

    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["status_file"] = path.name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    saved = merged[merged["status"] == "saved"].copy()

    duplicated = saved[saved.duplicated("task_id", keep=False)]
    if not duplicated.empty:
        print("WARNING: duplicated task IDs found. Keeping the most recently updated version.")
        saved["updated_at"] = saved["updated_at"].fillna("")
        saved = saved.sort_values("updated_at").drop_duplicates("task_id", keep="last")

    missing_masks = []
    for row in saved.itertuples():
        path = Path(str(row.mask_path))
        if not path.exists():
            missing_masks.append((row.task_id, str(path)))

    if missing_masks:
        preview = "\n".join(f"{task}: {path}" for task, path in missing_masks[:10])
        raise FileNotFoundError(f"Saved annotations with missing mask files:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Merged saved masks: {len(saved)}")
    print(f"Output: {args.output.resolve()}")
    print(saved["status_file"].value_counts().to_string())


if __name__ == "__main__":
    main()
