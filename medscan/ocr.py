"""
Text reader that keeps the recogniser's per-character probabilities.

RapidOCR runs PaddleOCR's PP-OCRv5 models in two stages: a detector finds text lines,
then a recogniser reads each line. The recogniser is a CNN + sequence model trained with
CTC: it slices the line image into ~40-80 thin vertical time steps and, for each step,
outputs a probability for every character in its alphabet (504 for the Latin model)
plus "blank" (nothing here). The usual output keeps only the most likely character per
step; we also keep the runners-up so medscan.ctc can re-read a field under rules
("4 letters then 8 digits") instead of trusting the single best guess.

Each item returned: (bbox [x0, y0, x1, y1], text, confidence, frames)
    frames = per time step, the top characters as [[char, prob], ...]; "" is blank.
"""

import cv2
import numpy as np

TOPK = 4


class Reader:
    def __init__(self, lang="latin", model="mobile"):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        self.eng = RapidOCR(params={"Rec.lang_type": LangRec(lang), "Rec.ocr_version": OCRVersion.PPOCRV5,
                                    "Rec.model_type": ModelType(model)})
        self.rec = self.eng._load_rec_model()
        self.chars = list(self.rec.postprocess_op.character)
        self.chars[0] = ""  # index 0 is the CTC blank

    def __call__(self, img, keep_frames=True):
        from rapidocr.utils.process_img import get_rotate_crop_image
        det = self.eng(img, use_rec=False, use_cls=False)
        if det.boxes is None or len(det.boxes) == 0:
            return []
        items = []
        for pts in det.boxes:
            crop = get_rotate_crop_image(img, np.asarray(pts, dtype=np.float32))
            h, w = crop.shape[:2]
            if h < 4 or w < 4:
                continue
            x = self.rec.resize_norm_img(crop, max(320 / 48, w / h))[None]
            probs = self.rec.session(x)[0]  # [T, C], softmax per time step
            text, conf = greedy_text(probs, self.chars)
            if not text.strip():
                continue
            frames = compact_frames(probs, self.chars) if keep_frames else None
            xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
            items.append(([min(xs), min(ys), max(xs), max(ys)], text, conf, frames))
        return items


def greedy_text(probs, chars):
    idx = probs.argmax(1)
    out, confs, prev = [], [], 0
    for t, i in enumerate(idx):
        if i != 0 and i != prev:
            out.append(chars[i])
            confs.append(float(probs[t, i]))
        prev = i
    return "".join(out), (float(np.mean(confs)) if confs else 0.0)


def compact_frames(probs, chars, k=TOPK, keep=0.005):
    """Top-k [[char, prob]] per time step; very sure steps keep only their winner (small cache)."""
    order = np.argsort(-probs, axis=1)[:, :k]
    frames = []
    for t in range(probs.shape[0]):
        row = []
        for i in order[t]:
            p = float(probs[t, i])
            if p < keep and row:
                break
            row.append([chars[i], round(p, 4)])
            if p > 0.995:
                break
        frames.append(row)
    return frames


def draw(img, items):
    for b, t, c, *_ in items:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 160, 255), 1)
    return img
