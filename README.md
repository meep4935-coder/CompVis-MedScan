# CompVis-MedScan

A learning project in computer vision, built in phases. The end goal: point a
camera at a piece of paper (or a label), follow it, and read a specific
sequence of letters and numbers from it.

| Phase | What it does | Main script |
|---|---|---|
| 1 | Find and box any piece of paper, keep an ID per sheet | `detect.py` |
| 2 | Outline the paper, track it, **capture a flattened scan** of its contents | `scan.py` |
| 3 | *(in progress)* Read the patient's **name, RAMQ (NAM) and date of birth** from anything in view | `medscan/` |

### Phase 3 so far: fake Quebec patient documents + RAMQ checker

All patients are **fictional**. Hospital names are real Quebec institutions used
as header text only; clinics and pharmacies are invented.

```bash
python generate_documents.py --n 300 --preview   # -> data/docs/ (image + JSON truth per doc)
python -m unittest tests.test_ramq                # RAMQ format / cross-check tests
```

Document types: hospital card, patient ID sticker, consultation note (with
sticker + handwriting), clinic prescription (patient info sometimes
handwritten), pharmacy bottle label (name only, no NAM), handwritten sticky note.
Each JSON stores the true patient and the box around every name / RAMQ / DOB on
the page, plus distractors (record no., phone, visit and expiry dates).

`medscan/ramq.py`: a RAMQ number is 4 letters (first 3 of the family name +
first initial) + 8 digits (YY, MM with +50 for women, DD, 2 admin digits), so
it cross-checks the name, sex and DOB read from the same document. It also
fixes typical OCR mix-ups using "letters first, digits after" (`7REM 8S55` -> `TREM 855…`).

### Choosing an OCR engine: `eval_ocr.py`

```bash
python eval_ocr.py --docs data/docs_eval            # clean pages
python eval_ocr.py --docs data/docs_eval --camera   # pages degraded like a webcam scan
```

Compares **RapidOCR** (PaddleOCR's PP-OCRv5 Latin models on ONNX Runtime; PaddlePaddle
itself has no Python 3.14 build, and ONNX is also how it would run on phones) with
**EasyOCR**. Scores against the JSON truth, printed vs handwritten: word recall,
exact read of each name / RAMQ / DOB, whether the true RAMQ is found on the page
(`find_ramq_loose` repairs letter/digit mix-ups), and seconds per page.

## How it works

```
generate_synthetic.py  ->  data/paper/ (images + outline labels)
train.py               ->  runs/paper_seg/weights/best.pt  (fine-tuned YOLO11n-seg)
scan.py                ->  outlines + track IDs + flattened scans in captures/scans/
```

1. **Synthetic data.** A COCO-pretrained YOLO has no "paper" class, so we need
   labelled examples. Instead of hand-labelling, we *render* them: random
   sheets (rectangles, A4/letter/sticky-note ratios, circles, torn edges,
   polygons; plain / lined / grid / text / printed / kraft textures, glossy
   glare) are perspective-warped onto random backgrounds with shadows,
   lighting gradients, blur, noise, and objects that occlude them. Since we
   drew every sheet, its exact outline is known for free.
   - **Labels** (class 1): product/pharmacy labels wrapped around bottles (with
     cylinder curvature + shine) or stuck on 3D boxes, often clear plastic.
   - **Hard negatives** (never labelled): curtains with folds, walls/doors,
     bare bottles and boxes, pens, blobs. These teach the model what paper is *not*.
2. **Transfer learning.** `train.py` fine-tunes `yolo11n-seg.pt`. The pretrained
   backbone already detects edges/corners/texture; training teaches the heads
   our two classes. Coming from MLPs: the backbone is a stack of conv layers
   (shared-weight MLPs sliding over the image); the detection head predicts,
   per grid cell, box offsets + class scores, and the segmentation head
   predicts mask coefficients that combine a few shared "prototype" masks into
   one mask per object.
3. **Tracking.** ByteTrack matches each frame's objects to the previous
   frame's by overlap (IoU) and a Kalman motion model, so each sheet keeps its ID.
4. **Corners + flattening** (`scan.py`). The outline is simplified to 4 corners
   (or the tightest rotated rectangle for round/torn paper), smoothed over
   frames, then a **perspective transform** (a 3x3 homography) maps them to a
   straight rectangle: the view a flatbed scanner would give.
5. **Best-frame capture.** For each tracked sheet, every frame's flattened view
   gets a score = sharpness (variance of the Laplacian) × size × glare
   penalty. Only sheets fully inside the frame and tracked for ≥ 5 frames are
   considered. The best view so far is saved, plus an enhanced greyscale copy.

### About the yolov5 repo

The original [ultralytics/yolov5](https://github.com/ultralytics/yolov5) repo is
script-based. Ultralytics now ships the same ideas as a pip package
(`ultralytics`), with newer models (YOLO11) and the old architecture still
available (`yolov5nu.pt`, boxes only).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python generate_synthetic.py --train 3000 --val 400 --preview   # ~5 min
python train.py --epochs 30                                      # ~2-3 h on CPU
python scan.py                                   # webcam: paper only
python scan.py --labels                          # also labels on bottles / boxes
python scan.py --source clip.mp4 --save-video    # video -> captures/scans/<time>/
python scan.py --source photo.jpg                # single image (or a folder)
```

Keys in the live window: **q** quit, **c** force-capture everything visible,
**s** save the raw frame (for retraining).

Output in `captures/scans/<date_time>/`:
- `paper_id3.jpg`: best flattened colour view of sheet #3
- `paper_id3_enhanced.jpg`: contrast-boosted greyscale (easier to read / OCR)
- `paper_id3_manual1.jpg`: anything captured with **c**

Training outputs useful plots in `runs/paper_seg/` (`results.png` loss/mAP
curves, `val_batch*_pred.jpg` predictions, `confusion_matrix.png`).

Known limits: a sheet held sideways is saved sideways (text orientation will be
handled with the reading step); curved labels are flattened as if they were flat.

## Testing on the MacBook (webcam) while developing on the desktop

The desktop does data generation and training; the Mac only runs the camera.
Trained models are committed in `models/`. On the Mac:
```bash
git clone https://github.com/meep4935-coder/CompVis-MedScan.git
cd CompVis-MedScan
bash mac/setup_mac.sh        # once (creates .venv, installs packages)
bash mac/run_scan.sh         # phase 2 (add --labels for labels)
bash mac/run_webcam.sh       # phase 1 box detector
```
Already cloned? `git pull` then `bash mac/run_scan.sh`.

macOS asks for camera permission for Terminal on first run (System Settings >
Privacy & Security > Camera). After a retrain on the desktop, run
`python make_bundle.py` (copies the new weights into `models/`), commit, push,
and `git pull` on the Mac. No git? The same command writes
`dist/CompV_bundle.zip` to copy over by USB/cloud.

**Feedback loop:** frames saved with **s** land in `captures/`. Copy the ones
showing things wrongly detected as paper (curtain, wall...) and containing *no*
paper or labels back into `data/backgrounds/` on the desktop. Then regenerate,
retrain, re-bundle.

## Making it work better on *your* camera

Synthetic-only training has a "sim-to-real gap". In order of impact:

1. **Real backgrounds**: 50–200 photos of your room/desk/curtain (*without*
   paper or labels) in `data/backgrounds/`, regenerate, retrain. Used for 70%
   of samples.
2. **Real labelled photos**: ~50–100 photos with paper, outlined with a polygon
   tool (e.g. [Label Studio](https://labelstud.io) or Roboflow, export YOLO
   segmentation format), added to `data/paper/images|labels/train`.
3. Tune `--conf` (raise to cut false positives, lower to catch more).

## Key metrics

- **Precision**: of the objects predicted, how many were right.
- **Recall**: of the real sheets, how many were found.
- **mAP50**: average precision when a prediction counts as correct at IoU ≥ 0.5.
- **mAP50-95**: same, averaged over stricter IoU thresholds (outline tightness).
- Reported twice for a seg model: **Box** (boxes) and **Mask** (outlines).
