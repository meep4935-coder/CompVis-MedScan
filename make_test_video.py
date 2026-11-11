"""
Make a test clip for `scan.py --read` without a webcam: a fake patient document
(from generate_documents.py) held in front of a background, drifting, tilting,
with motion blur, camera noise and JPEG-like softness, at 1280x720.

    python make_test_video.py --doc data/docs_eval/doc_00000.jpg --out clip.mp4
    python scan.py --source clip.mp4 --read --save-video [--auto-accept]
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

import generate_synthetic as gs

W, H = 1280, 720  # --hd switches to 1920x1080


def background(seed):
    random.seed(seed)
    np.random.seed(seed)
    gs.IMG_W, gs.IMG_H = W // 2, H // 2
    bg = gs.procedural_background()
    return cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="data/docs_eval/doc_00000.jpg")
    ap.add_argument("--out", default="captures/test_clip.mp4")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--height", type=float, default=0.8, help="page height as a fraction of the frame")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--hd", action="store_true", help="1920x1080 instead of 1280x720")
    args = ap.parse_args()
    global W, H
    if args.hd:
        W, H = 1920, 1080

    doc = cv2.imread(args.doc)
    if doc is None:
        raise SystemExit(f"cannot read {args.doc}")
    truth = Path(args.doc).with_suffix(".json")
    if truth.exists():
        p = json.loads(truth.read_text(encoding="utf-8"))["patient"]
        print(f"truth: {p['last']}, {p['first']} {p['middle']} | {p['ramq']} | {p['dob']}")
    bg = background(args.seed)
    rng = np.random.default_rng(args.seed)

    dh = args.height * H
    dw = dh * doc.shape[1] / doc.shape[0]
    if dw > 0.9 * W:  # wide documents (cards, labels): fit the width instead
        dw = 0.9 * W
        dh = dw * doc.shape[0] / doc.shape[1]
    src = np.float32([[0, 0], [doc.shape[1], 0], [doc.shape[1], doc.shape[0]], [0, doc.shape[0]]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    n = int(args.seconds * args.fps)
    prev_c = None
    for i in range(n):
        t = i / args.fps
        # slow drift + tilt, like a page held in a hand
        cx = W / 2 + 0.06 * W * math.sin(0.7 * t) + 0.02 * W * math.sin(2.3 * t)
        cy = H / 2 + 0.04 * H * math.sin(0.9 * t + 1)
        ang = math.radians(6 * math.sin(0.5 * t))
        tilt = 0.08 * math.sin(0.6 * t + 0.5)  # perspective: one side further away
        hw, hh = dw / 2, dh / 2
        pts = np.float32([[-hw * (1 - tilt), -hh], [hw * (1 + tilt), -hh * (1 + tilt)],
                          [hw * (1 + tilt), hh * (1 + tilt)], [-hw * (1 - tilt), hh]])
        rot = np.float32([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        dst = pts @ rot.T + np.float32([cx, cy])
        M = cv2.getPerspectiveTransform(src, dst)
        frame = bg.copy()
        warped = cv2.warpPerspective(doc, M, (W, H), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full(doc.shape[:2], 255, np.uint8), M, (W, H))
        shadow = cv2.GaussianBlur(cv2.warpAffine(mask, np.float32([[1, 0, 8], [0, 1, 10]]), (W, H)), (31, 31), 0)
        frame = (frame * (1 - 0.35 * shadow[..., None] / 255)).astype(np.uint8)
        frame[mask > 0] = warped[mask > 0]
        # motion blur proportional to how far the page moved since the last frame
        c = dst.mean(0)
        speed = 0 if prev_c is None else float(np.linalg.norm(c - prev_c))
        prev_c = c
        k = int(min(9, speed * 0.8))
        if k >= 2:
            kern = np.zeros((k, k), np.float32)
            kern[k // 2, :] = 1 / k
            frame = cv2.filter2D(frame, -1, kern)
        frame = cv2.GaussianBlur(frame, (0, 0), 0.8)
        frame = np.clip(frame + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    print(f"wrote {out} ({n} frames)")


if __name__ == "__main__":
    main()
