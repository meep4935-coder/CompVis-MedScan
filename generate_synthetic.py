"""
Synthetic dataset generator for "paper" (and "label") instance segmentation.

Why synthetic?  A pretrained YOLO model (trained on COCO) has no "paper" class,
so we must fine-tune it on labelled examples. Hand-labelling hundreds of photos
is slow, so we *render* images instead: we paste randomly shaped / coloured /
textured sheets of paper onto random backgrounds. Because we drew the paper
ourselves, we know exactly which pixels it covers -> labels come for free.

Classes:
    0 paper  - loose sheets, notes, cards, torn pieces
    1 label  - labels stuck on bottles / boxes (curved, often glossy)

Hard negatives (things that are NOT paper but look a bit like it) are drawn
too: curtains with folds, walls / doors, bare bottles and boxes, pens, etc.

Output layout (YOLO segmentation format):
    data/paper/images/{train,val}/xxxx.jpg
    data/paper/labels/{train,val}/xxxx.txt   # "class x1 y1 x2 y2 ..." polygon, normalised 0..1
    data/paper/paper.yaml                     # dataset config for ultralytics

Real photos dropped into data/backgrounds/ (desk, floor, room, curtain...,
WITHOUT any paper or labels in them) are used as backgrounds and make the
model generalise far better.

Usage:
    python generate_synthetic.py --train 3000 --val 400 --preview
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_W, IMG_H = 640, 480
CARD_P = 0.3  # share of "paper" objects drawn as plastic cards (health / ID / hospital cards)
ROOT = Path(__file__).parent
BG_DIR = ROOT / "data" / "backgrounds"
CLASS_NAMES = {0: "paper", 1: "label"}


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


def add_glare(img, mask, strength=None):
    """Specular highlight (shiny / laminated surface) inside `mask` (bool HxW)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img
    h, w = mask.shape
    layer = np.zeros((h, w), np.float32)
    c = (int(np.random.choice(xs)), int(np.random.choice(ys)))
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    axes = (max(3, int(span * random.uniform(0.05, 0.4))), max(2, int(span * random.uniform(0.02, 0.15))))
    cv2.ellipse(layer, c, axes, random.uniform(0, 180), 0, 360, 1.0, -1)
    k = int(max(3, span * 0.1)) | 1
    layer = cv2.GaussianBlur(layer, (k, k), 0) * (strength or random.uniform(0.5, 1.0))
    layer *= mask
    out = img.astype(np.float32)
    out = out * (1 - layer[..., None]) + 255 * layer[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def load_real_backgrounds():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not BG_DIR.exists():
        return []
    return [p for p in BG_DIR.rglob("*") if p.suffix.lower() in exts]


# ----------------------------------------------------------------------------
# Backgrounds + hard negatives
# ----------------------------------------------------------------------------
def draw_curtain(img):
    """Large fabric with vertical folds (the grey curtain that fooled phase 1)."""
    h, w = img.shape[:2]
    xa = random.randint(-w // 4, w // 2)
    xb = xa + int(w * random.uniform(0.4, 1.3))
    ya = 0 if random.random() < 0.7 else random.randint(0, h // 3)
    yb = h if random.random() < 0.8 else random.randint(h // 2, h)
    xa, xb = max(0, xa), min(w, xb)
    if xb - xa < 20 or yb - ya < 20:
        return
    cw = xb - xa
    color = np.array(rand_color(light_bias=random.choice([0.0, 0.3, 0.6])), np.float32)
    if random.random() < 0.5:  # greys / beiges are the common case
        g = random.randint(80, 220)
        color = np.array([g, g, g + random.randint(-10, 20)], np.float32)
    period = random.uniform(15, 70)
    x = np.arange(cw, dtype=np.float32)
    phase = 2 * np.pi * x / period + cv2.resize(np.random.rand(1, 6).astype(np.float32), (cw, 1))[0] * 4
    folds = (0.5 + 0.5 * np.sin(phase)) ** random.uniform(0.7, 2.0)
    shade = 0.55 + 0.45 * folds
    # folds drift slightly along the height
    rows = yb - ya
    drift = (np.linspace(0, 1, rows) * random.uniform(-0.15, 0.15))[:, None]
    shade2d = np.clip(shade[None, :] + drift * np.sin(phase)[None, :], 0.3, 1.1)
    fabric = color[None, None, :] * shade2d[..., None]
    fabric *= (0.93 + 0.07 * smooth_noise(rows, cw, 2, 1))
    img[ya:yb, xa:xb] = np.clip(fabric, 0, 255).astype(np.uint8)


def draw_wall_panel(img):
    """Door / wall / cabinet: big flat light area with frame lines running off-frame."""
    h, w = img.shape[:2]
    x0, x1 = sorted(random.sample(range(-w // 3, w + w // 3), 2))
    y0, y1 = -random.randint(0, h // 2), h + random.randint(0, h // 2)
    if x1 - x0 < w // 3:
        x1 = x0 + w // 2
    c = np.array(rand_color(light_bias=random.uniform(0.3, 0.9)), np.float32)
    region = np.zeros((h, w), np.uint8)
    cv2.rectangle(region, (x0, max(y0, 0)), (x1, min(y1, h)), 255, -1)
    m = region > 0
    g = np.linspace(random.uniform(0.75, 1.0), random.uniform(0.75, 1.0), w, dtype=np.float32)[None, :]
    img[m] = np.clip((c[None, None, :] * g[..., None] * np.ones((h, 1, 1)))[m], 0, 255).astype(np.uint8)
    frame_c = tuple(int(v * 0.6) for v in c)
    cv2.rectangle(img, (x0, y0), (x1, y1), frame_c, random.randint(3, 12))


def procedural_background():
    h, w = IMG_H, IMG_W
    kind = random.choice(["gradient", "noise", "stripes", "tiles", "clutter", "room"])
    base = np.array(rand_color(), np.float32)
    img = np.ones((h, w, 3), np.float32) * base

    if kind in ("gradient", "room"):
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

    img = np.clip(img, 0, 255).astype(np.uint8)
    if kind == "room":  # walls, doors, curtains
        for _ in range(random.randint(1, 2)):
            (draw_curtain if random.random() < 0.6 else draw_wall_panel)(img)
    if kind == "clutter" or random.random() < 0.5:
        draw_distractors(img, None, n=random.randint(3, 12))

    img = img.astype(np.float32) + np.random.randn(h, w, 3) * random.uniform(0, 8)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_background(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
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
    bg = None
    if real_bgs and random.random() < 0.7:
        bg = real_background(random.choice(real_bgs))
    if bg is None:
        return procedural_background()
    if random.random() < 0.2:
        draw_curtain(bg)
    return bg


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
        shade = 0.6 + 0.4 * smooth_noise(h, w, 64)[..., 0]
        img[m] = np.clip(np.array(color, np.float32) * shade[m][:, None], 0, 255).astype(np.uint8)
        if owner is not None:
            owner[m] = -1


# ----------------------------------------------------------------------------
# Textures
# ----------------------------------------------------------------------------
def draw_text_rows(tex, ink, x0, y0, x1, y1, gap=(10, 22), thick=(1, 3)):
    y = y0 + random.randint(5, 15)
    while y < y1 - 5:
        x = x0 + random.randint(3, 12)
        end = x1 - random.randint(3, 30)
        while x < end:
            word = random.randint(8, 45)
            cv2.line(tex, (x, y), (min(x + word, end), y), ink, random.randint(*thick))
            x += word + random.randint(4, 10)
        y += random.randint(*gap)


def paper_texture(tw, th, real_bgs):
    """Return a (th, tw, 3) uint8 texture for one sheet of paper."""
    kind = random.choices(
        ["plain", "lined", "grid", "text", "printed", "kraft"],
        weights=[3, 2, 1, 3, 1, 1],
    )[0]
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
        if random.random() < 0.6:  # handwriting on it
            draw_text_rows(tex, (random.randint(0, 120),) * 3, 0, gap, tw, th, (gap, gap), (1, 2))
    elif kind == "grid":
        gap = random.randint(8, 20)
        line_c = rand_color()
        for y in range(0, th, gap):
            cv2.line(tex, (0, y), (tw, y), line_c, 1)
        for x in range(0, tw, gap):
            cv2.line(tex, (x, 0), (x, th), line_c, 1)
    elif kind == "text":
        ink = (random.randint(0, 80),) * 3 if random.random() < 0.7 else rand_color()
        draw_text_rows(tex, ink, 5, 5, tw, th)
    elif kind == "printed":  # a photo / poster / magazine page
        if real_bgs and random.random() < 0.5:
            p = cv2.imread(str(random.choice(real_bgs)))
            if p is not None:
                tex = cv2.resize(p, (tw, th)).astype(np.float32)
        else:
            tex = smooth_noise(th, tw, random.choice([4, 8, 16]), 3) * 255
        if random.random() < 0.7:
            m = random.randint(3, 15)
            border = np.ones_like(tex) * 245
            border[m:-m, m:-m] = tex[m:-m, m:-m]
            tex = border
    elif kind == "kraft":
        tex = np.ones((th, tw, 3), np.float32) * np.array([90, 140, 180], np.float32)
        tex *= (0.85 + 0.3 * smooth_noise(th, tw, 2, 1))

    tex *= (0.95 + 0.1 * smooth_noise(th, tw, 3, 1))
    return np.clip(tex, 0, 255).astype(np.uint8)


def label_texture(tw, th):
    """Product / pharmacy label: brand block, text, nutrition-facts table, barcode."""
    base = rand_color(light_bias=random.choice([0.95, 0.8, 0.3, 0.0]))
    tex = np.ones((th, tw, 3), np.uint8) * np.array(base, np.uint8)
    ink = (random.randint(0, 60),) * 3 if sum(base) > 380 else (240, 240, 240)
    # split the label into 1-3 vertical panels with different content
    cuts = sorted(random.sample(range(1, 10), random.randint(0, 2)))
    xs = [0] + [int(tw * c / 10) for c in cuts] + [tw]
    for a, b in zip(xs[:-1], xs[1:]):
        kind = random.choice(["text", "table", "brand", "barcode"])
        if kind == "text":
            draw_text_rows(tex, ink, a, 0, b, th, (6, 14), (1, 2))
        elif kind == "table":  # nutrition facts
            m = 4
            cv2.rectangle(tex, (a + m, m), (b - m, th - m), ink, 1)
            y = m + random.randint(8, 16)
            cv2.line(tex, (a + m, y), (b - m, y), ink, random.randint(2, 5))
            while y < th - 10:
                y += random.randint(6, 11)
                cv2.line(tex, (a + m, y), (b - m, y), ink, 1)
                cv2.line(tex, (a + m + 3, y - 3), (a + m + random.randint(10, max(11, (b - a) // 2)), y - 3), ink, 1)
        elif kind == "brand":
            cv2.rectangle(tex, (a, random.randint(0, th // 3)), (b, random.randint(th // 2, th)), rand_color(), -1)
            draw_text_rows(tex, rand_color(light_bias=0.8), a, th // 3, b, th // 2 + 10, (8, 14), (2, 4))
        else:
            x = a + 5
            yb0, yb1 = int(th * 0.55), th - 5
            while x < b - 5:
                wbar = random.randint(1, 3)
                cv2.rectangle(tex, (x, yb0), (x + wbar - 1, yb1), (0, 0, 0), -1)
                x += wbar + random.randint(1, 3)
    return tex


# ----------------------------------------------------------------------------
# Objects: paper (class 0)
# ----------------------------------------------------------------------------
def paper_shape_mask(tw, th):
    """Mask (in texture coords) of the paper's outline: rectangle, circle, torn..."""
    mask = np.zeros((th, tw), np.uint8)
    shape = random.choices(["rect", "ellipse", "polygon", "torn", "rounded"], weights=[6, 1, 1, 2, 1])[0]
    if shape == "rect":
        mask[:] = 255
    elif shape == "ellipse":
        cv2.ellipse(mask, (tw // 2, th // 2), (tw // 2 - 1, th // 2 - 1), 0, 0, 360, 255, -1)
    elif shape == "polygon":
        k = random.randint(3, 7)
        ang = np.sort(np.random.uniform(0, 2 * np.pi, k))
        r = np.random.uniform(0.6, 1.0, k)
        pts = np.stack([tw / 2 + np.cos(ang) * r * tw / 2, th / 2 + np.sin(ang) * r * th / 2], 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    elif shape == "torn":  # jagged edge on 1-2 sides
        mask[:] = 255
        for _ in range(random.randint(1, 2)):
            n = 40
            xs = np.linspace(0, tw, n)
            ys = np.random.uniform(0, th * 0.15, n)
            pts = np.concatenate([np.stack([xs, ys], 1), [[tw, 0], [0, 0]]]).astype(np.int32)
            cv2.fillPoly(mask, [pts], 0)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
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
    aspect = random.choice([1.414, 1.294, 1.0, 1.6, 0.707, 0.773, random.uniform(0.4, 2.5)])
    size = random.uniform(0.12, 0.75) * min(w, h) * 1.4
    tw = int(np.clip(size, 30, 900))
    th = int(np.clip(size * aspect, 30, 900))

    tex = paper_texture(tw, th, real_bgs)
    mask = paper_shape_mask(tw, th)

    if random.random() < 0.2:  # crease / fold line
        x = random.randint(tw // 4, 3 * tw // 4)
        tex[:, x:] = (tex[:, x:].astype(np.float32) * random.uniform(0.8, 0.95)).astype(np.uint8)
    warp_onto(img, owner, idx, tex, mask, glare_p=0.2)


def warp_onto(img, owner, idx, tex, mask, glare_p=0.2, scale=(0.4, 1.0)):
    """Perspective-warp a flat texture (+ its shape mask) into the scene, with shadow,
    lighting gradient and optional glare. Pixels it covers get owner == idx.
    Returns the warped bool mask (or None if it ended up off-screen)."""
    h, w = img.shape[:2]
    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    cx, cy = random.uniform(-0.1, 1.1) * w, random.uniform(-0.1, 1.1) * h
    ang = random.uniform(0, 2 * np.pi)
    s = random.uniform(*scale)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (src - [tw / 2, th / 2]) * s
    dst = dst @ R.T + [cx, cy]
    dst += np.random.uniform(-0.12, 0.12, (4, 2)) * max(tw, th) * s
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    warped = cv2.warpPerspective(tex, M, (w, h), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    m = wmask > 127
    if m.sum() < 50:
        return None

    if random.random() < 0.7:  # drop shadow
        off = np.float32([[1, 0, random.randint(-15, 15)], [0, 1, random.randint(-15, 15)]])
        sh = cv2.warpAffine(wmask, off, (w, h))
        k = random.choice([9, 15, 25, 41])
        sh = cv2.GaussianBlur(sh, (k, k), 0).astype(np.float32) / 255 * random.uniform(0.2, 0.6)
        img[:] = (img.astype(np.float32) * (1 - sh[..., None])).astype(np.uint8)

    light = np.ones((h, w), np.float32)
    if random.random() < 0.7:
        gx, gy = np.random.uniform(-1, 1, 2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = (xx / w * gx + yy / h * gy)
        light = 1 + (g - g.mean()) * random.uniform(0.2, 0.6)
    paper = np.clip(warped.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)
    if random.random() < glare_p:  # glossy / laminated sheet
        paper = add_glare(paper, m)
        if random.random() < 0.3:
            paper = add_glare(paper, m)

    alpha = cv2.GaussianBlur(wmask, (3, 3), 0).astype(np.float32)[..., None] / 255
    img[:] = (paper * alpha + img * (1 - alpha)).astype(np.uint8)
    owner[m] = idx
    return m


# ----------------------------------------------------------------------------
# Plastic cards (health insurance / ID / hospital cards): counted as paper (class 0)
# Generic designs only: no real card's layout, logo or emblem.
# ----------------------------------------------------------------------------
def hologram(tex, mask):
    """Rainbow sheen of a security hologram / iridescent laminate: a soft diagonal band
    whose hue changes along it, blended on top."""
    th, tw = tex.shape[:2]
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    a = random.uniform(0, np.pi)
    d = (xx * np.cos(a) + yy * np.sin(a)) / max(tw, th)
    hue = ((d * random.uniform(2, 6) + random.random()) % 1.0 * 180).astype(np.uint8)
    hsv = np.dstack([hue, np.full_like(hue, 200), np.full_like(hue, 255)])
    rainbow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    c0 = random.uniform(-0.2, 1.2)
    band = np.exp(-((d - c0) / random.uniform(0.08, 0.35)) ** 2) * random.uniform(0.15, 0.45)
    band *= mask.astype(np.float32) / 255
    out = tex.astype(np.float32) * (1 - band[..., None]) + rainbow * band[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def card_texture(tw, th):
    """A made-up plastic card: coloured background with waves, a header band, a photo
    box with a face-like shape, rows of small text and sometimes a big number line."""
    base = np.array(random.choice([(170, 110, 40), (200, 150, 90), (90, 140, 40), (60, 60, 170),
                                   (235, 235, 235), (200, 200, 205), rand_color(0.3)]), np.float32)
    c2 = np.array(rand_color(0.4), np.float32)
    g = np.linspace(0, 1, tw, dtype=np.float32)[None, :, None]
    tex = base * (1 - g * 0.6) + c2 * (g * 0.6)
    tex = np.repeat(tex, th, 0)
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    for _ in range(random.randint(0, 4)):  # guilloche-like wavy security lines / bands
        f, ph, amp = random.uniform(1, 5), random.uniform(0, 6.3), random.uniform(0.05, 0.3)
        wave = np.abs(yy / th - (0.5 + amp * np.sin(xx / tw * f * 2 * np.pi + ph))) < random.uniform(0.01, 0.08)
        tex[wave] = tex[wave] * 0.75 + np.array(rand_color(0.5), np.float32) * 0.25
    tex = tex * (0.9 + 0.1 * smooth_noise(th, tw, 16))
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    if random.random() < 0.7:  # header band with a title
        hb = int(th * random.uniform(0.12, 0.25))
        cv2.rectangle(tex, (0, 0), (tw, hb), rand_color(), -1)
        draw_text_rows(tex, (255, 255, 255) if random.random() < 0.6 else (20, 20, 20),
                       int(tw * 0.05), int(hb * 0.2), int(tw * 0.8), hb, gap=(hb, hb + 1), thick=(2, 4))
    else:
        hb = 0
    ink = random.choice([(20, 20, 20), (60, 30, 10), (255, 255, 255)])
    photo = random.random() < 0.7
    px0 = int(tw * random.uniform(0.04, 0.08))
    if photo:  # photo box with a head-and-shoulders silhouette
        pw, ph = int(tw * 0.25), int(th * 0.55)
        py0 = min(th - ph - 2, hb + int(th * 0.08))
        cv2.rectangle(tex, (px0, py0), (px0 + pw, py0 + ph), rand_color(0.6), -1)
        skin = random.choice([(150, 180, 220), (110, 140, 190), (70, 100, 140), (40, 60, 90)])
        cv2.ellipse(tex, (px0 + pw // 2, py0 + int(ph * 0.42)), (pw // 4, int(ph * 0.22)), 0, 0, 360, skin, -1)
        cv2.ellipse(tex, (px0 + pw // 2, py0 + int(ph * 0.3)), (int(pw * 0.27), int(ph * 0.14)), 0, 180, 360,
                    rand_color() if random.random() < 0.3 else (30, 30, 40), -1)
        cv2.ellipse(tex, (px0 + pw // 2, py0 + ph), (int(pw * 0.45), int(ph * 0.3)), 0, 180, 360, rand_color(), -1)
        tx0 = px0 + pw + int(tw * 0.04)
    else:
        tx0 = px0
    draw_text_rows(tex, ink, tx0, hb + int(th * 0.05), tw - int(tw * 0.04), int(th * 0.8),
                   gap=(max(6, th // 12), max(7, th // 7)), thick=(1, 2))
    if random.random() < 0.6:  # the big number line (like an insurance / ID number)
        y = int(th * random.uniform(0.78, 0.9))
        draw_text_rows(tex, ink, px0, y - 6, int(tw * 0.75), y + 4, gap=(50, 60), thick=(2, 4))
    if random.random() < 0.25:  # chip or magnetic stripe
        if random.random() < 0.5:
            cx, cy = int(tw * 0.12), int(th * 0.45)
            cv2.rectangle(tex, (cx, cy), (cx + tw // 9, cy + th // 6), (60, 170, 210), -1)
        else:
            cv2.rectangle(tex, (0, int(th * 0.15)), (tw, int(th * 0.32)), (20, 20, 20), -1)
    return tex


def render_card(img, owner, idx):
    h, w = img.shape[:2]
    tw = int(np.clip(random.uniform(0.15, 0.65) * min(w, h) * 1.4, 40, 700))
    th = int(tw / 1.586)  # ID-1 card format, 85.6 x 54 mm
    tex = card_texture(tw, th)
    mask = np.zeros((th, tw), np.uint8)
    r = max(2, int(tw * random.uniform(0.03, 0.06)))  # rounded corners
    cv2.rectangle(mask, (r, 0), (tw - r, th), 255, -1)
    cv2.rectangle(mask, (0, r), (tw, th - r), 255, -1)
    for cx, cy in [(r, r), (tw - r, r), (r, th - r), (tw - r, th - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    if random.random() < 0.15:  # held upright / portrait
        tex, mask = np.ascontiguousarray(np.rot90(tex)), np.ascontiguousarray(np.rot90(mask))
    if random.random() < 0.45:
        tex = hologram(tex, mask)
    m = warp_onto(img, owner, idx, tex, mask, glare_p=0.7, scale=(0.5, 1.0))
    if m is not None and random.random() < 0.5:
        hold_with_fingers(img, owner, m)


def hold_with_fingers(img, owner, m):
    """A thumb / fingers over the card's edge (cards are usually held by hand)."""
    ys, xs = np.nonzero(m)
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ey, ex = np.nonzero(edge)
    if len(ex) == 0:
        return
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    skin = np.array(random.choice([(150, 180, 220), (120, 150, 200), (80, 110, 150), (50, 70, 100)]), np.float32)
    layer = np.zeros(m.shape, np.uint8)
    i = random.randrange(len(ex))
    cx, cy = int(ex[i]), int(ey[i])
    for k in range(random.randint(1, 3)):
        a = random.uniform(0, 180)
        cv2.ellipse(layer, (cx + k * int(span * 0.07), cy + k * int(span * 0.02)),
                    (max(4, int(span * 0.12)), max(3, int(span * 0.05))), a, 0, 360, 255, -1)
    f = layer > 0
    shade = 0.75 + 0.25 * smooth_noise(*m.shape, 32)[..., 0]
    img[f] = np.clip(skin * shade[f][:, None], 0, 255).astype(np.uint8)
    owner[f] = -1


# ----------------------------------------------------------------------------
# Objects: bottles / boxes, optionally with a label (class 1)
# ----------------------------------------------------------------------------
def place(img, owner, canvas, body, label, idx, transparent=False):
    """Rotate/scale/translate a locally drawn object (canvas + masks) into the scene."""
    h, w = img.shape[:2]
    ch, cw = body.shape
    target = random.uniform(0.25, 0.9) * h
    s = target / max(ch, cw)
    ang = random.uniform(-15, 15) if random.random() < 0.8 else random.choice([90, -90]) + random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, s)
    M[:, 2] += (random.uniform(0, w) - cw / 2, random.uniform(0, h) - ch / 2)
    can = cv2.warpAffine(canvas, M, (w, h))
    b = cv2.warpAffine(body, M, (w, h), flags=cv2.INTER_NEAREST) > 127
    l = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST) > 127 if label is not None else None
    out = img.astype(np.float32)
    if transparent:  # clear plastic: background shows through the body, not the label
        body_only = b & ~l if l is not None else b
        out[body_only] = out[body_only] * 0.6 + can[body_only] * 0.4
        if l is not None:
            out[l] = can[l]
    else:
        out[b] = can[b]
    img[:] = np.clip(out, 0, 255).astype(np.uint8)
    owner[b] = -1
    if l is not None and idx is not None:
        owner[l] = idx


def cylinder_shading(cw, x0, x1):
    cx, r = (x0 + x1) / 2, (x1 - x0) / 2
    t = np.clip((np.arange(cw, dtype=np.float32) - cx) / r, -1, 1)
    shade = 0.45 + 0.55 * np.sqrt(1 - t ** 2)
    t0, width = random.uniform(-0.6, 0.6), random.uniform(0.04, 0.12)
    spec = np.exp(-((t - t0) / width) ** 2) * random.uniform(0.2, 0.9)  # vertical shine stripe
    return t, shade, spec


def render_bottle(img, owner, idx, with_label):
    bw = random.randint(60, 220)
    bh = int(bw * random.uniform(1.2, 2.8))
    r = bw // 4
    pad = 6
    neck = int(bw * random.uniform(0.3, 0.6))
    cw, ch = bw + 2 * pad, bh + neck + 2 * pad
    x0, x1, top = pad, pad + bw, pad + neck
    body = np.zeros((ch, cw), np.uint8)
    cv2.rectangle(body, (x0, top + r), (x1, ch - pad), 255, -1)
    cv2.ellipse(body, ((x0 + x1) // 2, top + r), (bw // 2, r), 0, 180, 360, 255, -1)
    cap_w = int(bw * random.uniform(0.4, 0.8))
    cap = np.zeros_like(body)
    cv2.rectangle(cap, ((cw - cap_w) // 2, pad), ((cw + cap_w) // 2, top + r // 2), 255, -1)

    t, shade, spec = cylinder_shading(cw, x0, x1)
    color = np.array(random.choice([(40, 100, 170), (235, 235, 235), (160, 90, 40), (60, 140, 60), rand_color()]), np.float32)
    canvas = color[None, None, :] * shade[None, :, None] + 255 * spec[None, :, None]
    canvas = np.repeat(canvas, ch, 0)
    cap_c = np.array(rand_color(), np.float32)
    canvas[cap > 0] = (cap_c[None, :] * np.repeat(shade[None, :], ch, 0)[cap > 0][:, None])

    label = None
    if with_label:
        ly0 = random.randint(top + r + 3, top + r + max(4, (bh - r) // 4))
        ly1 = min(ch - pad - 3, ly0 + int((ch - pad - ly0) * random.uniform(0.4, 0.95)))
        lh = ly1 - ly0
        if lh > 10:
            tw = int(bw * random.uniform(1.5, 2.5))  # label wraps around: we see the front part
            tex = label_texture(tw, lh).astype(np.float32)
            # project the flat label onto the cylinder: x on screen -> arc position on label
            u = (np.arcsin(np.clip(t[x0:x1], -0.999, 0.999)) / (np.pi / 2) + 1) / 2
            u = (u * (tw / 2) + tw / 4 + random.uniform(-tw / 8, tw / 8)).astype(np.float32)
            map_x = np.repeat(u[None, :], lh, 0)
            map_y = np.repeat(np.arange(lh, dtype=np.float32)[:, None], x1 - x0, 1)
            lab = cv2.remap(tex, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            gloss = random.uniform(0.3, 1.0)
            lab = lab * shade[None, x0:x1, None] + 255 * spec[None, x0:x1, None] * gloss
            canvas[ly0:ly1, x0:x1] = lab
            label = np.zeros_like(body)
            label[ly0:ly1, x0:x1] = 255
            label &= body
    full = cv2.bitwise_or(body, cap)
    place(img, owner, np.clip(canvas, 0, 255).astype(np.uint8), full, label, idx,
          transparent=random.random() < 0.25)


def render_box(img, owner, idx, with_label):
    """3D box / container (often clear plastic) with a rectangular label on the front.
    The visible top + side faces are what make a box look different from a flat sheet."""
    bw, bh = random.randint(80, 300), random.randint(60, 260)
    d = int(min(bw, bh) * random.uniform(0.15, 0.5))  # depth shown in the oblique view
    flip = random.random() < 0.5  # side face on the left or right
    cw, ch = bw + d, bh + d
    color = np.array(rand_color(), np.float32)
    canvas = np.zeros((ch, cw, 3), np.float32)
    body = np.zeros((ch, cw), np.uint8)
    fx = d if flip else 0  # front face x offset
    front = np.array([[fx, d], [fx + bw, d], [fx + bw, ch], [fx, ch]], np.int32)
    top = np.array([[fx, d], [fx + bw, d], [fx + bw + (-d if flip else d), 0], [fx + (-d if flip else d), 0]], np.int32)
    sx = fx if flip else fx + bw
    side = np.array([[sx, d], [sx + (-d if flip else d), 0], [sx + (-d if flip else d), bh], [sx, ch]], np.int32)
    for poly, k in [(top, random.uniform(1.05, 1.25)), (side, random.uniform(0.45, 0.7)), (front, 1.0)]:
        m = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(m, [poly], 255)
        canvas[m > 0] = np.clip(color * k, 0, 255)
        body |= m
    edge_c = tuple(float(v) * 0.5 for v in color)
    for poly in (top, side, front):
        cv2.polylines(canvas, [poly], True, edge_c, 1)
    g = np.linspace(random.uniform(0.75, 1.0), random.uniform(0.75, 1.0), cw, dtype=np.float32)
    canvas *= g[None, :, None]
    label = None
    if with_label:
        lw, lh = int(bw * random.uniform(0.35, 0.9)), int(bh * random.uniform(0.25, 0.7))
        lx, ly = fx + random.randint(0, bw - lw), d + random.randint(0, bh - lh)
        canvas[ly:ly + lh, lx:lx + lw] = label_texture(lw, lh)
        label = np.zeros_like(body)
        label[ly:ly + lh, lx:lx + lw] = 255
    if random.random() < 0.6:  # shiny plastic / glossy label
        canvas = add_glare(np.clip(canvas, 0, 255).astype(np.uint8), body > 0).astype(np.float32)
    place(img, owner, np.clip(canvas, 0, 255).astype(np.uint8), body, label, idx,
          transparent=random.random() < 0.4)


# ----------------------------------------------------------------------------
# Scene + labels
# ----------------------------------------------------------------------------
def mask_to_polygon(mask):
    """Largest outer contour of a binary mask -> simplified (N,2) polygon, or None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def make_sample(real_bgs):
    img = make_background(real_bgs)
    owner = np.full(img.shape[:2], -1, np.int32)
    classes = []

    # bare bottles / boxes = hard negatives (the object itself is not paper)
    for _ in range(random.choices([0, 1, 2], weights=[5, 3, 1])[0]):
        (render_bottle if random.random() < 0.5 else render_box)(img, owner, None, with_label=False)

    n_papers = random.choices([0, 1, 2, 3], weights=[2, 10, 4, 1])[0]
    n_labels = random.choices([0, 1, 2], weights=[6, 3, 1])[0]
    objs = [0] * n_papers + [1] * n_labels
    random.shuffle(objs)
    for cls in objs:
        idx = len(classes)
        classes.append(cls)
        if cls == 0:
            if random.random() < CARD_P:
                render_card(img, owner, idx)
            else:
                render_paper(img, owner, idx, real_bgs)
        else:
            (render_bottle if random.random() < 0.6 else render_box)(img, owner, idx, with_label=True)

    if objs and random.random() < 0.4:  # pen / phone / hand on top
        draw_distractors(img, owner, n=random.randint(1, 3))

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        img = np.clip(img + np.random.randn(*img.shape) * random.uniform(2, 10), 0, 255).astype(np.uint8)

    labels, polys = [], []
    for idx, cls in enumerate(classes):
        poly = mask_to_polygon(owner == idx)
        if poly is None:
            continue
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        coords = " ".join(f"{x / IMG_W:.5f} {y / IMG_H:.5f}" for x, y in poly)
        labels.append(f"{cls} {coords}")
        polys.append((cls, poly))
    return img, labels, polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="draw outlines on a few samples into out/preview")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    real_bgs = load_real_backgrounds()
    print(f"Real background photos found: {len(real_bgs)} (put more in {BG_DIR})")

    for split, n in [("train", args.train), ("val", args.val)]:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels, polys = make_sample(real_bgs)
            name = f"{split}_{i:05d}"
            cv2.imwrite(str(args.out / "images" / split / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, random.randint(70, 95)])
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(labels))

            if args.preview and split == "val" and i < 24:
                (args.out / "preview").mkdir(exist_ok=True)
                vis = img.copy()
                for cls, poly in polys:
                    cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0) if cls == 0 else (0, 165, 255), 2)
                cv2.imwrite(str(args.out / "preview" / f"{name}.jpg"), vis)
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i + 1}/{n}")

    yaml = args.out / "paper.yaml"
    names = "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items())
    yaml.write_text(f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    print(f"Done. Dataset config: {yaml}")


if __name__ == "__main__":
    main()
