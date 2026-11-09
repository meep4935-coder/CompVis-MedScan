"""
Synthetic dataset generator for "paper" detection.

Why synthetic?  A pretrained YOLO model (trained on COCO) has no "paper" class,
so we must fine-tune it on labelled examples. Hand-labelling hundreds of photos
is slow, so we *render* images instead: we paste randomly shaped / coloured /
textured sheets of paper onto random backgrounds. Because we drew the paper
ourselves, we know exactly where it is -> labels come for free.

Output layout (YOLO format):
    data/paper/images/{train,val}/xxxx.jpg
    data/paper/labels/{train,val}/xxxx.txt   # "class cx cy w h" (normalised 0..1)
    data/paper/paper.yaml                     # dataset config for ultralytics

Real photos dropped into data/backgrounds/ (any jpg/png of desks, floors,
rooms, ...) are used as backgrounds and make the model generalise far better.

Usage:
    python generate_synthetic.py --train 2000 --val 300
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 640, 480
ROOT = Path(__file__).parent
BG_DIR = ROOT / "data" / "backgrounds"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def rand_color(light_bias=0.0):
    """Random BGR colour. light_bias in [0,1] pushes the colour towards white."""
    c = np.random.randint(0, 256, 3).astype(np.float32)
    c = c * (1 - light_bias) + 255 * light_bias
    return tuple(int(v) for v in c)


def smooth_noise(h, w, scale, channels=1):
    """Cheap 'cloudy' noise: tiny random image upscaled with bicubic interpolation."""
    sh, sw = max(2, h // scale), max(2, w // scale)
    small = np.random.rand(sh, sw, channels).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if big.ndim == 2:
        big = big[..., None]
    return np.clip(big, 0, 1)


def load_real_backgrounds():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not BG_DIR.exists():
        return []
    return [p for p in BG_DIR.rglob("*") if p.suffix.lower() in exts]


# ----------------------------------------------------------------------------
# Backgrounds
# ----------------------------------------------------------------------------
def procedural_background():
    h, w = IMG_H, IMG_W
    kind = random.choice(["gradient", "noise", "stripes", "tiles", "clutter"])
    base = np.array(rand_color(), np.float32)
    img = np.ones((h, w, 3), np.float32) * base

    if kind == "gradient":
        c2 = np.array(rand_color(), np.float32)
        t = np.linspace(0, 1, w if random.random() < 0.5 else h, dtype=np.float32)
        t = t[None, :, None] if t.size == w else t[:, None, None]
        img = img * (1 - t) + c2 * t
    elif kind == "noise":
        n = smooth_noise(h, w, random.choice([4, 16, 64]), 3)
        img = img * 0.5 + n * 255 * 0.5
    elif kind == "stripes":  # wood / fabric like
        freq = random.uniform(0.02, 0.2)
        ang = random.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        s = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * freq + smooth_noise(h, w, 32)[..., 0] * 6)
        img = img * (0.75 + 0.25 * s[..., None])
    elif kind == "tiles":
        size = random.randint(20, 120)
        c2 = np.array(rand_color(), np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx // size + yy // size) % 2).astype(bool)
        img[mask] = img[mask] * 0.5 + c2 * 0.5

    # Random clutter (distractor objects) on top of most backgrounds
    if kind == "clutter" or random.random() < 0.5:
        img = img.astype(np.uint8)
        draw_distractors(img, None, n=random.randint(3, 12))
        img = img.astype(np.float32)

    # global grain
    img += np.random.randn(h, w, 3) * random.uniform(0, 8)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_background(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    # random crop with the target aspect ratio, then resize
    scale = random.uniform(0.5, 1.0)
    cw, ch = int(w * scale), int(w * scale * IMG_H / IMG_W)
    if ch > h:
        ch, cw = h, int(h * IMG_W / IMG_H)
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (IMG_W, IMG_H))
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    return img


def make_background(real_bgs):
    if real_bgs and random.random() < 0.7:
        bg = real_background(random.choice(real_bgs))
        if bg is not None:
            return bg
    return procedural_background()


def draw_distractors(img, owner, n):
    """Non-paper objects (pens, cups, blobs...) so the model learns what paper is NOT.
    If `owner` is given, covered pixels are marked -1 (occludes paper underneath)."""
    h, w = img.shape[:2]
    for _ in range(n):
        layer = np.zeros((h, w), np.uint8)
        color = rand_color()
        kind = random.choice(["circle", "line", "blob", "rect"])
        if kind == "circle":
            cv2.circle(layer, (random.randint(0, w), random.randint(0, h)), random.randint(5, 50), 255, -1)
        elif kind == "line":  # pen / pencil / cable
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (p1[0] + random.randint(-200, 200), p1[1] + random.randint(-200, 200))
            cv2.line(layer, p1, p2, 255, random.randint(2, 12))
        elif kind == "blob":
            c = np.array([random.randint(0, w), random.randint(0, h)])
            pts = c + np.random.randint(-60, 60, (random.randint(5, 9), 2))
            cv2.fillPoly(layer, [cv2.convexHull(pts.astype(np.int32))], 255)
        else:  # small solid objects (phone, eraser, book spine) - darker / saturated
            c = (random.randint(0, w), random.randint(0, h))
            box = cv2.boxPoints((c, (random.randint(10, 80), random.randint(10, 160)), random.uniform(0, 180)))
            cv2.fillPoly(layer, [box.astype(np.int32)], 255)
            color = rand_color() if random.random() < 0.5 else tuple(int(v * 0.4) for v in rand_color())
        m = layer > 0
        # simple 3D-ish shading so distractors don't look like flat paper
        shade = 0.6 + 0.4 * smooth_noise(h, w, 64)[..., 0]
        img[m] = np.clip(np.array(color, np.float32) * shade[m][:, None], 0, 255).astype(np.uint8)
        if owner is not None:
            owner[m] = -1


# ----------------------------------------------------------------------------
# Paper rendering
# ----------------------------------------------------------------------------
def paper_texture(tw, th, real_bgs):
    """Return a (th, tw, 3) uint8 texture for one sheet of paper."""
    kind = random.choices(
        ["plain", "lined", "grid", "text", "printed", "kraft"],
        weights=[3, 2, 1, 2, 1, 1],
    )[0]
    # Most real paper is light, but we allow any colour
    base = rand_color(light_bias=random.choice([0.85, 0.7, 0.4, 0.0]))
    tex = np.ones((th, tw, 3), np.float32) * np.array(base, np.float32)

    if kind == "lined":
        gap = random.randint(10, 25)
        line_c = rand_color() if random.random() < 0.3 else (200, 150, 100)
        for y in range(gap * 2, th, gap):
            cv2.line(tex, (0, y), (tw, y), line_c, 1)
        if random.random() < 0.6:
            x = int(tw * 0.12)
            cv2.line(tex, (x, 0), (x, th), (80, 80, 220), 1)
    elif kind == "grid":
        gap = random.randint(8, 20)
        line_c = rand_color()
        for y in range(0, th, gap):
            cv2.line(tex, (0, y), (tw, y), line_c, 1)
        for x in range(0, tw, gap):
            cv2.line(tex, (x, 0), (x, th), line_c, 1)
    elif kind == "text":  # fake typed/handwritten text rows
        ink = (random.randint(0, 80),) * 3 if random.random() < 0.7 else rand_color()
        y = random.randint(10, 30)
        while y < th - 10:
            x = random.randint(8, 20)
            end = tw - random.randint(8, 40)
            while x < end:
                word = random.randint(10, 50)
                cv2.line(tex, (x, y), (min(x + word, end), y), ink, random.randint(1, 3))
                x += word + random.randint(4, 10)
            y += random.randint(10, 22)
    elif kind == "printed":  # a photo / poster / magazine page
        if real_bgs and random.random() < 0.5:
            p = cv2.imread(str(random.choice(real_bgs)))
            if p is not None:
                tex = cv2.resize(p, (tw, th)).astype(np.float32)
        else:
            tex = smooth_noise(th, tw, random.choice([4, 8, 16]), 3) * 255
        if random.random() < 0.7:  # white margin
            m = random.randint(3, 15)
            border = np.ones_like(tex) * 245
            border[m:-m, m:-m] = tex[m:-m, m:-m]
            tex = border
    elif kind == "kraft":  # brown / cardboard / recycled paper fibres
        tex = np.ones((th, tw, 3), np.float32) * np.array([90, 140, 180], np.float32)
        tex *= (0.85 + 0.3 * smooth_noise(th, tw, 2, 1))

    # paper grain
    tex *= (0.95 + 0.1 * smooth_noise(th, tw, 3, 1))
    return np.clip(tex, 0, 255).astype(np.uint8)


def paper_shape_mask(tw, th):
    """Mask (in texture coords) of the paper's outline: rectangle, circle, torn..."""
    mask = np.zeros((th, tw), np.uint8)
    shape = random.choices(["rect", "ellipse", "polygon", "torn", "rounded"], weights=[6, 1, 1, 1, 1])[0]
    if shape == "rect":
        mask[:] = 255
    elif shape == "ellipse":
        cv2.ellipse(mask, (tw // 2, th // 2), (tw // 2 - 1, th // 2 - 1), 0, 0, 360, 255, -1)
    elif shape == "polygon":  # triangle, pentagon, cut-out shape
        k = random.randint(3, 7)
        ang = np.sort(np.random.uniform(0, 2 * np.pi, k))
        r = np.random.uniform(0.6, 1.0, k)
        pts = np.stack([tw / 2 + np.cos(ang) * r * tw / 2, th / 2 + np.sin(ang) * r * th / 2], 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    elif shape == "torn":  # rectangle with a jagged edge
        mask[:] = 255
        n = 40
        xs = np.linspace(0, tw, n)
        ys = np.random.uniform(0, th * 0.15, n)
        pts = np.concatenate([np.stack([xs, ys], 1), [[tw, 0], [0, 0]]]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 0)
    else:  # rounded corners (sticky notes, cards)
        r = int(min(tw, th) * random.uniform(0.05, 0.2))
        cv2.rectangle(mask, (r, 0), (tw - r, th), 255, -1)
        cv2.rectangle(mask, (0, r), (tw, th - r), 255, -1)
        for cx, cy in [(r, r), (tw - r, r), (r, th - r), (tw - r, th - r)]:
            cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def render_paper(img, owner, idx, real_bgs):
    """Warp a textured paper onto `img`. Pixels it covers get owner == idx."""
    h, w = img.shape[:2]
    # Paper size on screen (relative to image), and aspect (A4, letter, square, note...)
    aspect = random.choice([1.414, 1.294, 1.0, 1.6, 0.707, 0.773, random.uniform(0.4, 2.5)])
    size = random.uniform(0.12, 0.75) * min(w, h) * 1.4
    tw = int(np.clip(size, 30, 900))
    th = int(np.clip(size * aspect, 30, 900))

    tex = paper_texture(tw, th, real_bgs)
    mask = paper_shape_mask(tw, th)

    # crease / fold line
    if random.random() < 0.2:
        x = random.randint(tw // 4, 3 * tw // 4)
        tex[:, x:] = (tex[:, x:].astype(np.float32) * random.uniform(0.8, 0.95)).astype(np.uint8)

    # --- perspective warp: random rotation, scale, tilt, position (can be partly off-screen)
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    cx, cy = random.uniform(-0.1, 1.1) * w, random.uniform(-0.1, 1.1) * h
    ang = random.uniform(0, 2 * np.pi)
    s = random.uniform(0.4, 1.0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (src - [tw / 2, th / 2]) * s
    dst = dst @ R.T + [cx, cy]
    dst += np.random.uniform(-0.12, 0.12, (4, 2)) * max(tw, th) * s  # perspective jitter
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    warped = cv2.warpPerspective(tex, M, (w, h), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    m = wmask > 127
    if m.sum() < 50:
        return

    # --- drop shadow (blurred, offset, darkens background)
    if random.random() < 0.7:
        off = np.float32([[1, 0, random.randint(-15, 15)], [0, 1, random.randint(-15, 15)]])
        sh = cv2.warpAffine(wmask, off, (w, h))
        k = random.choice([9, 15, 25, 41])
        sh = cv2.GaussianBlur(sh, (k, k), 0).astype(np.float32) / 255 * random.uniform(0.2, 0.6)
        img[:] = (img.astype(np.float32) * (1 - sh[..., None])).astype(np.uint8)

    # --- lighting gradient across the sheet
    light = np.ones((h, w), np.float32)
    if random.random() < 0.7:
        gx, gy = np.random.uniform(-1, 1, 2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (xx / w * gx + yy / h * gy)
        light = 1 + (g - g.mean()) * random.uniform(0.2, 0.6)
    paper = np.clip(warped.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)

    # slightly soft edge (anti-aliasing)
    alpha = cv2.GaussianBlur(wmask, (3, 3), 0).astype(np.float32)[..., None] / 255
    img[:] = (paper * alpha + img * (1 - alpha)).astype(np.uint8)
    owner[m] = idx


# ----------------------------------------------------------------------------
# Scene + labels
# ----------------------------------------------------------------------------
def make_sample(real_bgs):
    img = make_background(real_bgs)
    owner = np.full(img.shape[:2], -1, np.int32)  # which paper is visible at each pixel

    n_papers = random.choices([0, 1, 2, 3], weights=[1, 12, 4, 1])[0]  # ~5% negatives
    for i in range(n_papers):
        render_paper(img, owner, i, real_bgs)

    # things lying ON the paper (pen, phone, hand...) -> occlusion
    if n_papers and random.random() < 0.4:
        draw_distractors(img, owner, n=random.randint(1, 3))

    # camera effects
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        img = np.clip(img + np.random.randn(*img.shape) * random.uniform(2, 10), 0, 255).astype(np.uint8)

    labels = []
    for i in range(n_papers):
        ys, xs = np.nonzero(owner == i)
        if len(xs) < 150:  # too small / fully hidden -> not labelled
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        cx, cy = (x0 + x1) / 2 / IMG_W, (y0 + y1) / 2 / IMG_H
        bw, bh = (x1 - x0) / IMG_W, (y1 - y0) / IMG_H
        labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="draw boxes on a few samples into out/preview")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    real_bgs = load_real_backgrounds()
    print(f"Real background photos found: {len(real_bgs)} (put more in {BG_DIR})")

    for split, n in [("train", args.train), ("val", args.val)]:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = make_sample(real_bgs)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(70, 95)])
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(labels))

            if args.preview and split == "val" and i < 16:
                (args.out / "preview").mkdir(exist_ok=True)
                vis = img.copy()
                for l in labels:
                    _, cx, cy, bw, bh = map(float, l.split())
                    p1 = (int((cx - bw / 2) * IMG_W), int((cy - bh / 2) * IMG_H))
                    p2 = (int((cx + bw / 2) * IMG_W), int((cy + bh / 2) * IMG_H))
                    cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                cv2.imwrite(str(args.out / "preview" / f"{name}.jpg"), vis)
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{n}")

    yaml = args.out / "paper.yaml"
    yaml.write_text(f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: paper\n")
    print(f"Done. Dataset config: {yaml}")


if __name__ == "__main__":
    main()
