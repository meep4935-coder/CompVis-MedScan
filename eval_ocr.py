"""
Compare OCR engines on the fake Quebec documents (generate_documents.py output).

    python eval_ocr.py --docs data/docs_eval                 # both engines
    python eval_ocr.py --engines rapid --limit 20             # quick run

Engines
    rapid  RapidOCR = PaddleOCR's PP-OCRv5 models (Latin, mobile size) run with ONNX Runtime.
           (PaddlePaddle itself has no build for Python 3.14; ONNX is also how it would run on phones.)
    easy   EasyOCR (French + English).

Scores, split by printed vs handwritten (from the JSON ground truth)
    word recall   share of the page's true words that the engine read somewhere on the page
    field read    for each name / RAMQ / DOB on the page: did the engine read it exactly
                  (strict = accents and all; loose = ignoring accents and case)
    RAMQ found    the true RAMQ appears among RAMQ-looking reads (find_ramq_loose, which also
                  repairs letter/digit mix-ups); 'false RAMQ' counts wrong candidates
    s/page        seconds per page on this CPU

Raw OCR results are cached in <docs>/ocr_<engine>.json so the field-extraction step can
reuse them without re-running OCR.
"""

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2

from medscan.ramq import find_ramq_loose


# ---------- engines: each returns [(x0, y0, x1, y1), text, confidence] ----------

def load_rapid():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    eng = RapidOCR(params={"Rec.lang_type": LangRec.LATIN, "Rec.ocr_version": OCRVersion.PPOCRV5,
                           "Rec.model_type": ModelType.MOBILE})

    def run(img):
        r = eng(img)
        if r.boxes is None:
            return []
        return [(_bbox(b), t, float(s)) for b, t, s in zip(r.boxes, r.txts, r.scores)]
    return run


def load_easy():
    import easyocr
    reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)

    def run(img):
        return [(_bbox(b), t, float(s)) for b, t, s in reader.readtext(img)]
    return run


ENGINES = {"rapid": load_rapid, "easy": load_easy}


def camera_like(img, seed):
    """Make a clean render look like a page flattened from a webcam frame (scan.py output):
    lower resolution, slight blur, sensor noise, uneven light, JPEG. Returns (image, scale)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.35, 0.6)  # a page ~600-1000 px tall instead of 1754
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.1))
    h, w = out.shape[:2]
    gx = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.9, 1.1), w)[None, :, None]
    gy = np.linspace(rng.uniform(0.8, 1.05), rng.uniform(0.75, 1.0), h)[:, None, None]
    out = out.astype(np.float32) * gx * gy + rng.normal(0, rng.uniform(3, 9), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 85))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), s


def _bbox(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


# ---------- text helpers ----------

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def squash(s, loose=False):
    """Compare without spaces; loose also ignores accents and case."""
    s = re.sub(r"\s+", "", s)
    return strip_accents(s).upper() if loose else s


def words(s):
    return re.findall(r"[\wÀ-ÿ'\-]+", s.lower())


def overlap(a, b):
    """Area of intersection / area of b (b = the ground-truth box)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def text_near(ocr, box, pad=6):
    """OCR text whose boxes cover part of a ground-truth box, in reading order."""
    b = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
    hits = [(o[0], o[1]) for o in ocr if overlap(o[0], b) > 0.05]
    hits.sort(key=lambda h: (round(h[0][1] / 20), h[0][0]))
    return " ".join(t for _, t in hits)


# ---------- scoring ----------

def score_doc(truth, ocr, stats):
    page_text = " ".join(t for _, t, _ in ocr)
    page_words = defaultdict(int)
    for w in words(page_text):
        page_words[w] += 1

    for line in truth["lines"]:
        kind = "hand" if line["hand"] else "print"
        avail = dict(page_words)
        for w in words(line["text"]):
            stats[f"words_{kind}"] += 1
            if avail.get(w, 0) > 0:
                stats[f"words_ok_{kind}"] += 1
                avail[w] -= 1

    for f in truth["fields"]:
        kind = "hand" if f["hand"] else "print"
        key = f"{f['field'] if f['field'] in ('ramq', 'dob') else 'name'}_{kind}"
        got = text_near(ocr, f["bbox"])
        stats[f"n_{key}"] += 1
        if f["field"] == "ramq":
            ok_strict = squash(f["text"]) in squash(got)
            ok_loose = truth["patient"]["ramq"] in find_ramq_loose(got)
        else:
            ok_strict = squash(f["text"]) in squash(got)
            ok_loose = squash(f["text"], True) in squash(got, True)
        stats[f"strict_{key}"] += ok_strict
        stats[f"loose_{key}"] += ok_loose

    true_r = truth["patient"]["ramq"]
    on_page = any(f["field"] == "ramq" for f in truth["fields"])
    cands = find_ramq_loose(page_text)
    if on_page:
        stats["ramq_pages"] += 1
        stats["ramq_found"] += true_r in cands
    stats["ramq_false"] += sum(c != true_r for c in cands)


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   -  "


def report(name, s, n_docs, secs):
    print(f"\n=== {name}  ({n_docs} docs, {secs / max(1, n_docs):.1f} s/page) ===")
    print(f"word recall      print {pct(s['words_ok_print'], s['words_print'])}   "
          f"hand {pct(s['words_ok_hand'], s['words_hand'])}")
    print(f"{'field read':16} {'strict':>14} {'loose':>14}   (n)")
    for fld in ("name", "ramq", "dob"):
        for kind in ("print", "hand"):
            k = f"{fld}_{kind}"
            if s[f"n_{k}"]:
                print(f"  {k:14} {pct(s['strict_' + k], s['n_' + k]):>14} "
                      f"{pct(s['loose_' + k], s['n_' + k]):>14}   ({s['n_' + k]})")
    print(f"RAMQ found on page {pct(s['ramq_found'], s['ramq_pages'])} of {s['ramq_pages']} pages; "
          f"false RAMQ candidates: {s['ramq_false']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs_eval")
    ap.add_argument("--engines", default="rapid,easy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="re-run OCR even if cached")
    ap.add_argument("--camera", action="store_true",
                    help="degrade pages like a webcam scan first (low-res, blur, noise, light, JPEG)")
    ap.add_argument("--save-camera", type=int, default=0, help="also save N degraded pages to look at")
    args = ap.parse_args()

    docs = Path(args.docs)
    jsons = sorted(docs.glob("doc_*.json"))
    if args.limit:
        jsons = jsons[:args.limit]
    truths = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in jsons}

    summary = {}
    for name in args.engines.split(","):
        cache_file = docs / f"ocr_{name}{'_camera' if args.camera else ''}.json"
        cache = {} if args.no_cache or not cache_file.exists() else json.loads(cache_file.read_text(encoding="utf-8"))
        todo = [k for k in truths if k not in cache]
        secs = cache.get("_secs", 0.0) * len(truths)
        if todo:
            run = ENGINES[name]()
            t0 = time.time()
            for i, k in enumerate(todo, 1):
                img, s = cv2.imread(str(docs / f"{k}.jpg")), 1.0
                if args.camera:
                    img, s = camera_like(img, seed=int(k.split("_")[1]))
                    if i <= args.save_camera:
                        cv2.imwrite(str(docs / f"{k}_camera.jpg"), img)
                # boxes back to the clean page's coordinates, where the ground truth lives
                cache[k] = [([v / s for v in b], t, c) for b, t, c in run(img)]
                print(f"\r{name}: {i}/{len(todo)}", end="", flush=True)
            secs = time.time() - t0
            cache["_secs"] = secs / len(todo)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            secs = cache["_secs"] * len(truths)

        stats = defaultdict(int)
        by_type = defaultdict(lambda: defaultdict(int))
        for k, truth in truths.items():
            score_doc(truth, cache[k], stats)
            score_doc(truth, cache[k], by_type[truth["type"]])
        report(name, stats, len(truths), secs)
        print("  by document type (word recall all / RAMQ found):")
        for t, s in sorted(by_type.items()):
            wr = pct(s["words_ok_print"] + s["words_ok_hand"], s["words_print"] + s["words_hand"])
            print(f"    {t:13} {wr}   {pct(s['ramq_found'], s['ramq_pages'])}")
        summary[name] = dict(stats)

    (docs / f"ocr_summary{'_camera' if args.camera else ''}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
