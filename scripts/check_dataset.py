
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = {
    "train": 31047,
    "validation": 1739,
    "test": 2343,
}

REQUIRED_KEYS = {
    "left_hand",
    "right_hand",
    "pose",
    "face",
    "text",
    "timestamps",
}

EXPECTED_SHAPES = {
    "left_hand": (21, 3),
    "right_hand": (21, 3),
    "pose": (25, 3),
    "face": (40, 3),
}


def check_file(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS - set(data.files)
            if missing:
                return f"Missing keys: {sorted(missing)}"

            arrays = {}
            for key, expected_tail in EXPECTED_SHAPES.items():
                arr = data[key]

                if arr.ndim != 3 or arr.shape[1:] != expected_tail:
                    return f"{key} has unexpected shape {arr.shape}"

                if arr.shape[0] == 0:
                    return f"{key} has zero frames"

                if not np.issubdtype(arr.dtype, np.number):
                    return f"{key} is not numeric"

                if not np.isfinite(arr).all():
                    return f"{key} contains NaN or infinity"

                arrays[key] = arr

            frame_count = arrays["pose"].shape[0]
            for key, arr in arrays.items():
                if arr.shape[0] != frame_count:
                    return f"{key} frame count differs from pose"

            timestamps = np.asarray(data["timestamps"])
            if timestamps.ndim != 1 or len(timestamps) != frame_count:
                return "Timestamp count does not match frame count"

            if not np.issubdtype(timestamps.dtype, np.number):
                return "Timestamps are not numeric"

            if not np.isfinite(timestamps).all():
                return "Timestamps contain NaN or infinity"

            if len(timestamps) > 1 and not np.all(
                np.diff(timestamps) > 0
            ):
                return "Timestamps are not strictly increasing"

            text = data["text"]
            if not isinstance(text.item(), str) or not text.item().strip():
                return "Text annotation is empty or invalid"

        return None

    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main():
    total_checked = 0
    total_failed = 0

    for split, expected_count in SPLITS.items():
        folder = ROOT / "data" / "processed" / split
        files = sorted(folder.glob("*.npz")) if folder.exists() else []

        print(f"\n=== {split.upper()} ===")
        print(f"Expected files: {expected_count}")
        print(f"Found files:    {len(files)}")

        if len(files) != expected_count:
            print("WARNING: File count does not match expected count.")

        split_failed = 0

        for index, path in enumerate(files, start=1):
            error = check_file(path)

            if error:
                split_failed += 1
                print(f"FAIL: {path.name} -> {error}")

            if index % 500 == 0 or index == len(files):
                print(f"Checked {index}/{len(files)} files")

        split_passed = len(files) - split_failed
        total_checked += len(files)
        total_failed += split_failed

        print(f"Passed: {split_passed}")
        print(f"Failed: {split_failed}")

    print("\n=== FINAL SUMMARY ===")
    print(f"Total files checked: {total_checked}")
    print(f"Total failed files:  {total_failed}")
    print(f"Total passed files:  {total_checked - total_failed}")


if __name__ == "__main__":
    main()
