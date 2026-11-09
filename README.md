# CompVis-MedScan

A learning project in computer vision, built in phases.

## Phase 1: Paper detection & tracking

Goal: find and box **any piece of paper** (any shape, colour, texture) on any
background, in images, video, or a live webcam, and keep a stable ID per sheet.

Later phases will refine this toward more precise medical-scan work.

## How it works

```
generate_synthetic.py  ->  data/paper/ (images + YOLO labels)
train.py               ->  runs/paper/weights/best.pt  (fine-tuned YOLO11n)
detect.py              ->  boxes + track IDs on webcam / video / images
```

1. **Synthetic data.** A COCO-pretrained YOLO has no "paper" class, so we need
   labelled examples. Instead of hand-labelling, we *render* them: random
   sheets (rectangles, A4/letter/sticky-note ratios, circles, torn edges,
   polygons; plain / lined / grid / text / printed / kraft textures) are
   perspective-warped onto random backgrounds with shadows, lighting
   gradients, blur, noise, and distractor objects that occlude them. Since we
   drew every sheet, the bounding box is exact and free.
2. **Transfer learning.** `train.py` fine-tunes `yolo11n.pt` (2.6M params). The
   pretrained backbone already detects edges/corners/texture; training just
   teaches the head what "paper" looks like. Coming from MLPs: the backbone is
   a stack of conv layers (shared-weight MLPs sliding over the image), and the
   head predicts, per grid cell, box offsets + an objectness/class score.
3. **Tracking.** `detect.py` uses ByteTrack: each frame's boxes are matched to
   the previous frame's by overlap (IoU) and a Kalman motion model, so each
   sheet keeps its ID.

### About the yolov5 repo

The original [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repo is
script-based (`train.py` / `detect.py` inside the clone). Ultralytics now ships
the same ideas as a pip package (`ultralytics`), with newer models (YOLO11) and
the old architecture still available as `yolov5nu.pt`. To compare:
`python train.py --model yolov5nu.pt --name paper_v5`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python generate_synthetic.py --train 2000 --val 300 --preview   # ~3 min
python train.py --epochs 30                                      # ~80 min on CPU
python detect.py --source 0                                      # webcam, press q to quit
python detect.py --source my_video.mp4 --save                    # -> runs/detect_out/
python detect.py --source photo.jpg
```

Training outputs useful plots in `runs/paper/` (`results.png` loss/mAP curves,
`val_batch*_pred.jpg` predictions, `confusion_matrix.png`).

## Testing on the MacBook (webcam) while developing on the desktop

The desktop does data generation and training; the Mac only runs `detect.py`.

The trained model is committed at `models/paper_best.pt`, so the easiest route
is git. On the Mac:
```bash
git clone https://github.com/meep4935-coder/CompVis-MedScan.git
cd CompVis-MedScan
bash mac/setup_mac.sh        # once (creates .venv, installs packages)
bash mac/run_webcam.sh       # q = quit, s = save a raw frame to captures/
```
macOS asks for camera permission for Terminal on first run (System Settings >
Privacy & Security > Camera). After a retrain on the desktop, run
`python make_bundle.py` (copies the new weights into `models/`), commit, push,
and `git pull` on the Mac.

No git on the Mac? `python make_bundle.py` also writes `dist/CompV_bundle.zip`
(code + model) that you can copy over by USB/cloud and unzip.

**Feedback loop:** frames saved with `s` on the Mac land in `captures/`. Copy
the ones *without* paper back into `data/backgrounds/` on the desktop, and keep
the ones *with* paper for labelling. Then regenerate, retrain, re-bundle.

## Making it work better on *your* camera

Synthetic-only training has a "sim-to-real gap". In order of impact:

1. **Real backgrounds**: drop 50–200 photos of your desk/floor/room (*without*
   paper) into `data/backgrounds/`, regenerate, retrain. Used for 70% of samples.
2. **Real labelled photos**: take ~50–100 photos with paper, label them (e.g.
   [Label Studio](https://labelstud.io) or Roboflow, export YOLO format) and
   add them to `data/paper/images|labels/train`.
3. Tune `--conf` in `detect.py` (raise to cut false positives, lower to catch more).

## Key metrics

- **Precision**: of the boxes predicted, how many were paper.
- **Recall**: of the real sheets, how many were found.
- **mAP50**: average precision when a box counts as correct at IoU ≥ 0.5.
- **mAP50-95**: same, averaged over stricter IoU thresholds (box tightness).
