"""
Detect and track paper in a webcam stream, video, image, or folder of images.

Detection = "where is paper in THIS frame" (bounding box + confidence).
Tracking  = linking detections across frames so each sheet keeps the same ID
            (uses ByteTrack, which matches boxes frame-to-frame by overlap and
            a motion model).

Usage:
    python detect.py --source 0                  # webcam 0, live window
    python detect.py --source clip.mp4 --save    # video -> runs/detect_out/
    python detect.py --source photo.jpg          # single image
    python detect.py --source data/paper/images/val --save --no-show
Keys in the live window:
    q  quit
    s  save the raw (un-annotated) frame to captures/  -> bring these back to the
       training machine as real backgrounds or photos to label
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAPTURE_DIR = ROOT / "captures"
# models/ holds "released" weights (portable, committed); runs/ is local training output
DEFAULT_WEIGHTS = next((p for p in [ROOT / "models" / "paper_best.pt",
                                    ROOT / "runs" / "paper" / "weights" / "best.pt"] if p.exists()),
                       ROOT / "models" / "paper_best.pt")


def draw(frame, result, fps=None):
    boxes = result.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
        conf = float(boxes.conf[i])
        tid = int(boxes.id[i]) if boxes.id is not None else None
        label = f"paper {conf:.2f}" + (f"  id {tid}" if tid is not None else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def run_images(model, paths, args, out_dir):
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        vis = draw(img, res)
        print(f"{p.name}: {len(res.boxes)} paper(s)")
        if args.save:
            cv2.imwrite(str(out_dir / p.name), vis)
        if args.show:
            cv2.imshow("paper", vis)
            if cv2.waitKey(0 if len(paths) == 1 else 500) & 0xFF == ord("q"):
                break


def run_stream(model, source, args, out_dir):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    writer = None
    t_prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # persist=True keeps the tracker state between calls -> stable IDs
        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6))
        t_prev = now
        raw = frame.copy()
        vis = draw(frame, res, fps)

        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
            writer.write(vis)
        if args.show:
            cv2.imshow("paper", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(exist_ok=True)
                p = CAPTURE_DIR / f"cap_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
                cv2.imwrite(str(p), raw)
                print(f"saved {p}")
    cap.release()
    if writer:
        writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--source", default="0", help="webcam index, video file, image, or folder")
    ap.add_argument("--conf", type=float, default=0.4, help="minimum confidence to keep a box")
    ap.add_argument("--imgsz", type=int, default=416, help="use the same size you trained with")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out_dir = ROOT / "runs" / "detect_out"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    if src.is_dir():
        run_images(model, sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS), args, out_dir)
    elif src.suffix.lower() in IMG_EXTS:
        run_images(model, [src], args, out_dir)
    else:
        run_stream(model, args.source, args, out_dir)
    cv2.destroyAllWindows()
    if args.save:
        print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
