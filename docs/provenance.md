# Source and verification record

## Source

Organized on 2026-11-12 from the user's downloaded `“rock_pick_autolabel (1).ipynb”的副本`.

- Original size: 14,406,490 bytes; 8 cells.
- Original SHA-256: `d0c6743c3961fe58bf8dae0bd5db1f5a8d89e19824d4c8c57294216d299c75e1`.
- Original file remains unchanged. The raw notebook and its embedded imagery are not committed.
- The source uses pretrained `cotracker3_offline` for annotation. It contains no YOLO training run, trained detector checkpoint, or accuracy metrics.

## Historical evidence in the original outputs

The source recorded a Tesla T4 runtime, 25 discovered videos, and a `red-25` tracking run producing 68 saved frames from 542 input frames. A different retained cell output exported 60 image/label pairs for `red-24`. These outputs came from different execution counts; they do not establish a single end-to-end run, a complete dataset size, or model accuracy. They are source observations, not verification of the organized copy.

## Changes in the organized copy

- Removed all cell outputs, execution counts, widget state, and transient Colab cell metadata.
- Corrected instructions to describe Drive selection, sampled tracking, per-clip ZIPs, and the annotation-only scope.
- Made project/output paths explicit; added supplemental dependencies and common video extensions.
- Pinned CoTracker source to `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`; the source used an unpinned hub reference. This pin is for the organized copy, not a recovered source-run revision.
- Preserved keypoint order and defaults for sampling, black-border cropping, 640×480 output, bounding-box padding, and final five-visible-point filtering.
- Used actual per-axis resize factors for tracking coordinate conversion and rejected nonfinite/out-of-bounds keypoints in label generation.
- Added empty-video, duplicate-output, write-success, and review-before-export checks.
- Changed frame rejection from deletion to a local quarantine folder. Source videos remain unchanged.
- Replaced `val: train/images` with `val: val/images`. The latter is intentionally absent from an annotation batch until independent splits are prepared.
- Used the full clip name in ZIP filenames and limited rendered previews to 24 sampled frames.

## Verification scope

On 2026-11-12, all **7 CPU tests passed** using Python 3.12 in the existing local MRover environment. They cover notebook syntax/metadata, crop geometry and padding, invalid keypoints, malformed review input, synthetic black-border estimation, rejected-pair isolation, and current-clip-only ZIP export. Notebook v4 schema validation, Ruff `F,E9` checks, local Markdown link checks, credential-pattern scanning, and original-source SHA-256 verification also passed. Ruff and nbformat ran in a separate temporary validation environment; project dependencies were not changed.

A fresh Colab run remains necessary to verify dependency installation, Drive authorization, GPU model loading, JavaScript clicks, widgets, real-video tracking, and Roboflow import. No GPU tracking or training was launched during repository organization. No raw images/videos or pretrained/trained weight files are included.
