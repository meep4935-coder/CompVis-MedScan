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

import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

TOPK = 4
DICTS = Path(__file__).parent / "dicts"


def pick_backend():
    """'onnxruntime' when installed, else 'opencv' (OpenCV's DNN module runs the same ONNX
    model files: same output to ~1e-6, similar speed, and it installs everywhere OpenCV does,
    e.g. Intel Macs or Python versions onnxruntime has no build for).
    Force one with the environment variable MEDSCAN_OCR_BACKEND=opencv|onnxruntime."""
    forced = os.environ.get("MEDSCAN_OCR_BACKEND")
    if forced:
        return forced
    return "onnxruntime" if importlib.util.find_spec("onnxruntime") else "opencv"


def _use_opencv_engine():
    """Make RapidOCR run its models with cv2.dnn instead of onnxruntime."""
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
    from rapidocr.utils.log import logger

    class CvInferSession(InferSession):
        def __init__(self, cfg):
            model_path = cfg.get("model_path", None)
            if model_path is None:  # same default models / download as RapidOCR's ONNX engine
                root = Path(cfg.get("model_root_dir"))
                root.mkdir(parents=True, exist_ok=True)
                info = self.get_model_url(FileInfo(engine_type=cfg.engine_type, ocr_version=cfg.ocr_version,
                                                   task_type=cfg.task_type, lang_type=cfg.lang_type,
                                                   model_type=cfg.model_type))
                model_path = root / Path(info["model_dir"]).name
                DownloadFile.run(DownloadFileInput(file_url=info["model_dir"], sha256=info["SHA256"],
                                                   save_path=model_path, logger=logger))
            self.model_path = Path(model_path)
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

        def __call__(self, x):
            self.net.setInput(np.ascontiguousarray(x, dtype=np.float32))
            return self.net.forward()

        # the character list lives in the ONNX metadata, which cv2 can't read: kept in medscan/dicts
        def _dict(self):
            return DICTS / f"{self.model_path.stem}.txt"

        def have_key(self, key="character"):
            return key == "character" and self._dict().exists()

        def get_character_list(self, key="character"):
            return self._dict().read_bytes().decode("utf-8").splitlines()

    import rapidocr.ch_ppocr_cls.main as cls_m
    import rapidocr.ch_ppocr_det.main as det_m
    import rapidocr.ch_ppocr_rec.main as rec_m
    for m in (det_m, rec_m, cls_m):
        m.get_engine = lambda engine_type: CvInferSession


class Reader:
    def __init__(self, lang="latin", model="mobile", backend=None):
        self.backend = backend or pick_backend()
        if self.backend == "opencv":
            _use_opencv_engine()
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
