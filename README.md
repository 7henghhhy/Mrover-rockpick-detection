# MRover Rock-pick Detection

A Colab workflow for preparing five-keypoint rock-pick annotations with **CoTracker3**, for downstream **YOLO Pose** training. Click the five landmarks once per video, track them through sampled frames, inspect the results, and export YOLO-format image/label pairs to Google Drive.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/7henghhhy/Mrover-rockpick-detection/blob/main/notebooks/rock_pick_autolabel.ipynb)

This repository currently contains the annotation workflow. The supplied notebook does not train YOLO and contains no trained detector weights or accuracy evaluation.

## Run in Colab

1. Open the notebook above and select **Runtime → Change runtime type → T4 GPU**.
2. Put your videos in a Drive folder with this layout, then edit `PROJECT_DIR` in Cell 1:

   ```text
   MyDrive/rockpick/
   ├── Videos/       # .mov, .mp4, .m4v; extensions are case-insensitive
   └── Annotated/    # created automatically for exported ZIPs
   ```

3. Run Cells **1–2** once to install dependencies, mount Drive, and load the pretrained tracker.
4. For each clip, run **3 → 4 → 5 → 6 → 7**. Cell 3 requires clicking **Load selected video** after selecting a clip. In Cell 4, choose a clear `CLICK_FRAME` divisible by `TRACK_STRIDE` and click the landmarks in the order below.
5. Inspect previews, enter incorrect frame numbers in `BAD`, and run Cell 6. It also rejects frames with fewer than five visible keypoints. Cell 7 saves **only this clip** as `Annotated/rock_pick_labels_<clip>.zip`. Export before selecting the next clip.

| Index | Keypoint |
| --- | --- |
| 0 | `blunt-end` |
| 1 | `metal-intersection` |
| 2 | `sharp-end` |
| 3 | `metal-handle-intersection` |
| 4 | `handle` |

Class ID is `0` (`rock_pick`). Preserve this keypoint order in Roboflow and every downstream model.

## Processing and output

The notebook estimates a stable black-border crop from sampled frames, tracks at a reduced resolution, and saves a **640 × 480** crop centered on the visible pick. Smaller inputs receive black padding. Coordinates are translated into that saved crop and normalized for YOLO labels.

| Setting | Default | Meaning |
| --- | --- | --- |
| `TRACK_STRIDE` | 8 | Track every eighth decoded frame |
| `EVERY` | 8 | Save every eighth decoded frame; must be a multiple of tracking stride |
| `MAX_SIDE` | 640 | Longest tracking-image side |
| `PAD` | 0.08 | Bounding-box padding relative to keypoint extent |
| `MIN_VISIBLE` | 3 | Minimum points retained during initial export |
| Cell 6 review | 5 visible points | Final acceptance rule before ZIP export |

Each per-clip ZIP contains:

```text
train/images/<clip>_f00000.jpg
train/labels/<clip>_f00000.txt
data.yaml
```

Each label has **20 fields**: class ID, normalized box `(cx, cy, width, height)`, then five `(x, y, visibility)` triples. The tracker supplies provisional labels, so human inspection remains necessary.

**Create disjoint train/validation/test splits by original recording or capture session before training.** Adjacent frames or segments from the same source recording should stay together. The ZIP is an annotation batch, not a ready-to-train split: its `data.yaml` points validation to `val/images`, which must be created during dataset preparation. The original notebook pointed training and validation to the same images; this copy removes that leakage.

## Files and reproducibility

- [`notebooks/rock_pick_autolabel.ipynb`](notebooks/rock_pick_autolabel.ipynb): cleaned, output-free Colab notebook.
- [`docs/workflow.md`](docs/workflow.md): 中文运行说明、review、OOM 处理及训练交接。
- [`docs/provenance.md`](docs/provenance.md): source evidence, changes, historical output limits, and verification.
- [`requirements-colab.txt`](requirements-colab.txt): supplemental notebook dependencies; keep Colab's GPU-compatible PyTorch.
- [`tests/test_notebook.py`](tests/test_notebook.py): CPU checks for geometry, labels, review, and ZIP export.

The tracker code is pinned to CoTracker commit `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`. Python package versions remain determined by Colab; this is not a fully locked environment. Cell 1 prints runtime versions. CoTracker weights are downloaded by the upstream hub loader, not stored here.

Run CPU checks in an environment containing NumPy and OpenCV:

```bash
python -m unittest discover -s tests -v
```

## Limits

- GPU tracking and the click interface require Colab; the organized copy still needs a fresh end-to-end Colab run.
- Only one rock pick is annotated per clip. Five visible landmarks are required for final acceptance, so occluded examples are excluded by default.
- Black-border removal is heuristic. Inspect dark scenes and crops that truncate part of the pick.
- Boxes are derived from landmark extent with padding; they are not manually verified object outlines.
- The preview renders up to 24 evenly spaced saved frames. Review all exported images before using them as training ground truth.
- CoTracker3 offline mode still loads the sampled clip into GPU memory. For OOM, trim long clips or increase `TRACK_STRIDE` and adjust `EVERY`/`CLICK_FRAME` accordingly.
- Local intermediate files disappear with the Colab runtime. ZIPs exported to Drive persist. Existing per-clip files/ZIPs are guarded against accidental overwrite; use a fresh `OUT` folder for a rerun.

No raw videos, annotations, private project weights, or embedded notebook outputs are published in this repository.

## References

- [CoTracker repository and license terms](https://github.com/facebookresearch/co-tracker)
- [CoTracker pinned hub loader](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/hubconf.py)
- [Ultralytics YOLO Pose dataset format](https://docs.ultralytics.com/datasets/pose/)

Upstream code, pretrained weights, and datasets retain their respective license and access terms.
