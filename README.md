# How2Sign Sign Language Recognition

A team project using the How2Sign dataset and MediaPipe Holistic
landmarks. Data preparation is complete; sequence-model training,
ablation experiments, evaluation, and model export remain future stages.

> **Current status:** 35,129 processed landmark files exist across
> train, validation, and test. The structural/numeric integrity check
> and annotation consistency check each reported 35,129 passed and 0
> failed.

## Project overview

The pipeline samples video frames at approximately **5 FPS**, extracts
left-hand, right-hand, upper-body pose, and selected face landmarks with
the MediaPipe Holistic Tasks API, normalizes coordinates relative to the
shoulder midpoint and shoulder distance, and stores each clip in a
compressed NumPy `.npz` file. Each file also contains the corresponding
sentence and metadata.

## Current progress

  Split           Raw clips   Annotation rows   Processed `.npz`
  ------------ ------------ ----------------- ------------------
  Train              31,047            31,165             31,047
  Validation          1,739             1,741              1,739
  Test                2,343             2,357              2,343
  **Total**      **35,129**        **35,263**         **35,129**

Completed: dataset organization, annotation inspection, MediaPipe setup,
extraction for all three splits, channel separation, structural/numeric
checks, and annotation consistency checks.

Not yet completed: sequence model training, ablations, BLEU-1 to BLEU-4,
ROUGE-L, WER, ONNX/TorchScript export, and an end-to-end inference
application.

## Repository contents

The original video files have been removed from this working project.
The annotation CSV files remain in `data/raw/`. The processed landmark
files are retained in `data/processed/` and are intended to be shared
through Git LFS.

``` text
SignLanguageProject/
├── data/
│   ├── raw/
│   │   ├── train/how2sign_train.csv
│   │   ├── validation/how2sign_val.csv
│   │   └── test/how2sign_test.csv
│   └── processed/
│       ├── train/                  31,047 .npz files
│       ├── validation/             1,739 .npz files
│       └── test/                   2,343 .npz files
├── models/
│   └── holistic_landmarker.task
├── scripts/
│   ├── extract_landmarks.py
│   ├── extract_landmarks_backup.py
│   ├── test_holistic.py
│   └── dataset/annotation checking scripts
├── outputs/                        currently empty; reserved for experiment outputs
├── requirements.txt
├── .gitattributes                  configures Git LFS for *.npz
├── .gitignore
└── README.md
```

The local `sign/` Python virtual environment is intentionally excluded
from version control. Each teammate should create their own environment.

## Dataset and licensing

The dataset used is **How2Sign**, noted by the project team as licensed
under **CC BY-NC 4.0**. Review the official dataset terms and
attribution requirements before sharing original videos, annotations, or
derived data outside the team. Do not use commercially unless the
applicable rights permit it.

The original video files are not included in this repository. The
annotation CSV files remain in the raw-data split folders. Each teammate
should obtain any required source videos through the approved source and
follow the dataset's applicable terms.

The MediaPipe task model at `models/holistic_landmarker.task` is
included because the extraction scripts refer to that path. Verify and
follow the model's redistribution terms when sharing the repository.

## Processed `.npz` format

`T` is the variable number of sampled frames in a clip.

  -------------------------------------------------------------------------
  Key                       Type / shape            Meaning
  ------------------------- ----------------------- -----------------------
  `left_hand`               float32 `(T, 21, 3)`    Left-hand landmarks

  `right_hand`              float32 `(T, 21, 3)`    Right-hand landmarks

  `pose`                    float32 `(T, 25, 3)`    Upper-body pose
                                                    landmarks, indices
                                                    0--24

  `face`                    float32 `(T, 40, 3)`    Selected face landmarks

  `text`                    scalar string           Ground-truth sentence

  `sentence_id`             scalar string           Annotation
                                                    sentence/segment ID

  `timestamps`              float32 `(T,)`          Sample timestamps

  `frame_count`             scalar integer          Number of sampled
                                                    frames

  `original_frame_count`    scalar integer          Original video frame
                                                    count

  `original_fps`            scalar float            Original video FPS

  `clip_duration_seconds`   scalar float            Original clip duration
  -------------------------------------------------------------------------

Feature sizes per frame after flattening XYZ: - Hands only: 126 -
Hands + pose: 201 - Hands + pose + selected face: 321

**Pose-count discrepancy:** the current extraction files contain **25
pose landmarks (indices 0--24)**. An earlier ablation outline mentioned
33 pose landmarks. Confirm the intended configuration with the project
lead before training/reporting ablations. Do not describe the current
files as containing 33 pose landmarks.

If a hand or face is not detected in a sampled frame, its landmarks are
zero-filled. The current format does not store a separate visibility
mask.

## Git LFS and processed data

All processed `.npz` files are configured to use **Git Large File
Storage (Git LFS)**. The repository's `.gitattributes` file contains the
tracking rule for `*.npz`.

Before cloning or working with the processed data, install Git and Git
LFS. After installing Git LFS, run:

``` powershell
git lfs install
```

Clone the repository using its GitHub HTTPS URL. Git LFS should retrieve
the tracked `.npz` objects as part of the clone. If the large files are
not present, check that Git LFS is installed and that the account has
access to the repository and its LFS objects.

The processed dataset is approximately **1.7 GB** in the current project
folder. GitHub LFS storage and bandwidth are subject to
account/organization quotas and plan limits. Confirm that the
organization has sufficient LFS storage and bandwidth before pushing or
inviting teammates to download the complete dataset.

## Setup (Windows 11 / PowerShell)

The original environment used Python 3.11.9. Create your own
environment; do not copy another machine's `sign/` directory.

From the project root:

``` powershell
py -3.11 -m venv sign
.\sign\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Current `requirements.txt`:

``` text
numpy
opencv-python
pandas
mediapipe
tqdm
```

Ensure the model exists at `models/holistic_landmarker.task`.

## Run validation checks

Run from the project root with the environment activated:

``` powershell
python scripts/check_annotations.py
```

This compares each processed file's `sentence_id` and `text` against the
corresponding split annotation table. Recorded completed result:

``` text
Total files checked: 35129
Total failures:      0
Total passed:        35129
```

The full structural/numeric checker was also run across all splits and
recorded 35,129 passed, 0 failed. Confirm the actual checker filename in
`scripts/` before running it; project notes previously referred to
`scripts/check_dataset.py`.

## Run landmark extraction

Main script: `scripts/extract_landmarks.py`.

**Important:** the completed working copy was last configured for the
**test** split. Before running, inspect and confirm these settings:

``` python
VIDEO_DIR = ROOT / "data" / "raw" / "test" / "raw_videos"
CSV_PATH = ROOT / "data" / "raw" / "test" / "how2sign_test.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "test"
SAMPLE_FPS = 5.0
MAX_VIDEOS = None
```

The original video folders are not included in this repository, so
extraction requires obtaining the source videos through the approved
dataset source and placing them in the expected folder structure.

To process train or validation, change all three split-specific paths
(`VIDEO_DIR`, `CSV_PATH`, `OUTPUT_DIR`) together. Then run:

``` powershell
python scripts/extract_landmarks.py
```

The script skips outputs that already exist, logs per-video errors to
`extraction_errors.txt` inside the output split folder, and creates a
fresh MediaPipe detector per video to avoid a previously observed
frame-size/smoothing-state error. Check paths and dataset availability
before running.

`extract_landmarks_backup.py` is a backup copy; use the main script
unless the team has a specific reason to use the backup.
`test_holistic.py` is a utility for testing the MediaPipe Holistic task
model.

## Load processed data

``` python
from pathlib import Path
import numpy as np

sample_path = Path("data/processed/train") / "YOUR_SAMPLE_FILE.npz"

with np.load(sample_path, allow_pickle=False) as sample:
    left_hand = sample["left_hand"]       # (T, 21, 3)
    right_hand = sample["right_hand"]     # (T, 21, 3)
    pose = sample["pose"]                 # (T, 25, 3)
    face = sample["face"]                 # (T, 40, 3)
    text = str(sample["text"].item())
    sentence_id = str(sample["sentence_id"].item())
    timestamps = sample["timestamps"]     # (T,)
```

Example channel concatenation:

``` python
hands = np.concatenate([left_hand, right_hand], axis=1).reshape(len(left_hand), -1)  # 126
hands_pose = np.concatenate([left_hand, right_hand, pose], axis=1).reshape(len(left_hand), -1)  # 201
full = np.concatenate([left_hand, right_hand, pose, face], axis=1).reshape(len(left_hand), -1)  # 321
```

A model pipeline still needs a variable-length batching/padding strategy
and text tokenization.

## Team responsibilities and next steps

**Member 1 --- Data Pipeline and Landmark Modularity:** dataset
organization, annotation checks, MediaPipe extraction, separate landmark
channels, data integrity verification, and handover documentation. This
stage is complete.

**Member 2 --- ML Architecture and Ablation Lead:** sequence model
design/training, agreed ablation experiments, BLEU-1 through BLEU-4,
ROUGE-L and WER evaluation, and ONNX or TorchScript export.

Suggested next tasks: 1. Resolve the 25-versus-33 pose landmark
specification. 2. Confirm the GitHub organization has sufficient Git LFS
storage and bandwidth for the dataset. 3. Implement dataset loading,
batching/padding, and text tokenization. 4. Train baseline and ablation
configurations. 5. Evaluate on the held-out test split and document
metrics. 6. Export and test the final model.

## Git and data hygiene

The current repository configuration is intended to: - Track processed
`*.npz` files with Git LFS. - Keep annotation CSV files and project
scripts under version control. - Include
`models/holistic_landmarker.task`, subject to its redistribution
terms. - Exclude the local `sign/` virtual environment, Python caches,
editor settings, and temporary log files. - Exclude original video
folders; original videos are not part of the current project folder.

Do not commit passwords, API keys, tokens, credentials, or unrelated
personal files.

Keep the repository private during development unless public release is
approved and dataset/model redistribution rights have been checked. The
team should confirm that sharing the processed landmark files and
annotation CSV files is permitted under the applicable dataset terms.

Use GitHub Issues to track work and Pull Requests to review changes
before merging.
