
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SPLITS = {
    "train": {
        "csv": ROOT / "data/raw/train/how2sign_train.csv",
        "processed": ROOT / "data/processed/train",
    },
    "validation": {
        "csv": ROOT / "data/raw/validation/how2sign_val.csv",
        "processed": ROOT / "data/processed/validation",
    },
    "test": {
        "csv": ROOT / "data/raw/test/how2sign_test.csv",
        "processed": ROOT / "data/processed/test",
    },
}


def load_annotations(csv_path):
    annotations = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            segment_id = row["SENTENCE_ID"].strip()
            sentence = row["SENTENCE"].strip()

            if segment_id in annotations:
                print(f"WARNING: Duplicate ID in {csv_path.name}: {segment_id}")

            annotations[segment_id] = sentence

    return annotations


def main():
    total_checked = 0
    total_failed = 0

    for split, paths in SPLITS.items():
        print(f"\n=== {split.upper()} ===")

        if not paths["csv"].exists():
            print(f"ERROR: CSV not found: {paths['csv']}")
            continue

        annotations = load_annotations(paths["csv"])
        files = sorted(paths["processed"].glob("*.npz"))

        print(f"Unique CSV annotation IDs: {len(annotations)}")
        print(f"Processed files:           {len(files)}")

        split_failed = 0

        for index, path in enumerate(files, start=1):
            try:
                with np.load(path, allow_pickle=False) as data:
                    saved_id = str(data["sentence_id"].item()).strip()
                    saved_text = str(data["text"].item()).strip()

                if saved_id not in annotations:
                    raise ValueError(f"Sentence ID not found in CSV: {saved_id}")

                expected_text = annotations[saved_id]

                if saved_text != expected_text:
                    raise ValueError(
                        f"Text mismatch for ID {saved_id}: "
                        f"saved={saved_text!r}, CSV={expected_text!r}"
                    )

            except Exception as exc:
                split_failed += 1
                print(f"FAIL: {path.name}: {exc}")

            if index % 500 == 0 or index == len(files):
                print(f"Checked {index}/{len(files)} files")

        total_checked += len(files)
        total_failed += split_failed

        print(f"Passed: {len(files) - split_failed}")
        print(f"Failed: {split_failed}")

    print("\n=== FINAL SUMMARY ===")
    print(f"Total files checked: {total_checked}")
    print(f"Total failures:      {total_failed}")
    print(f"Total passed:        {total_checked - total_failed}")


if __name__ == "__main__":
    main()
