"""
Fine-tune a pretrained YOLO *segmentation* model to find "paper" and "label".

Segmentation = for every object, which exact pixels belong to it (an outline),
not just a box. We need the outline to find the sheet's corners and flatten it.

Transfer learning: we start from weights trained on COCO (80 everyday classes).
The backbone already knows edges, corners, textures; we only need to teach the
head our new classes. That's why a few thousand synthetic images are enough.

Usage:
    python train.py                               # defaults below (CPU friendly)
    python train.py --model yolo11n.pt --name paper_det   # boxes only (phase 1 style)
    python train.py --epochs 50 --imgsz 640 --device 0     # if you have a CUDA GPU
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "paper" / "paper.yaml"))
    ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416, help="smaller = faster on CPU")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="'cpu' or GPU index like '0'")
    ap.add_argument("--name", default="paper_seg")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from runs/<name>/weights/last.pt")
    args = ap.parse_args()

    if args.resume:  # e.g. after a reboot: same settings, picks up at the next epoch
        last = ROOT / "runs" / args.name / "weights" / "last.pt"
        YOLO(last).train(resume=True)
        print(f"\nBest weights: {ROOT / 'runs' / args.name / 'weights' / 'best.pt'}")
        return

    model = YOLO(args.model)  # downloads the pretrained weights the first time
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        workers=4,
        patience=10,       # stop early if val mAP stops improving
        # Augmentations on top of our synthetic variety
        hsv_h=0.05, hsv_s=0.7, hsv_v=0.5,
        degrees=10, perspective=0.0005,
        fliplr=0.5, flipud=0.2,
        mosaic=1.0,
        close_mosaic=5,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}\nTry it: python scan.py --weights {best} --source 0")


if __name__ == "__main__":
    main()
