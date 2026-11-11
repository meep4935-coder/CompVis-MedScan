"""
Phase 2: outline, track and CAPTURE what is on each piece of paper.

Pipeline per frame
    1. Segment   - YOLO-seg gives each sheet's exact outline (a polygon), not just a box.
    2. Track     - ByteTrack links outlines across frames -> each sheet keeps an ID.
    3. Corners   - simplify the outline to its 4 corners (or the tightest rotated
                   rectangle for round / torn shapes). Smoothed over frames to reduce jitter.
    4. Flatten   - a perspective warp maps those 4 corners to a straight rectangle:
                   the "scanner" view, as if the camera looked straight down.
    5. Capture   - for every tracked sheet, keep the BEST view seen so far
                   (sharp, big, little glare, fully in frame) and save it.

Saved to captures/scans/<session>/:
    paper_id3.jpg            flattened colour view (best so far)
    paper_id3_enhanced.jpg   contrast-boosted greyscale, easier to read / OCR later

With --read (Phase 3), each sheet is also READ in the background: name, RAMQ and DOB are
voted on across views; when confident, a bar shows the patient record to accept or reject.
Accepted records are saved as patient_N.json next to the captures.

Usage:
    python scan.py                          # webcam 0
    python scan.py --read                   # + read name / RAMQ / DOB, one-tap confirm
    python scan.py --labels                 # also track labels on bottles / boxes
    python scan.py --source clip.mp4 --save-video
    python scan.py --source photo.jpg       # or a folder of images
Keys (live window): q quit | c force-capture everything visible | s save raw frame
                    Enter accept the proposed patient | Backspace or r reject it
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from medscan.camera import advice as camera_advice, is_black, open_camera

ROOT = Path(__file__).parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "paper", 1: "label"}
COLORS = {0: (0, 220, 0), 1: (0, 165, 255)}
DEFAULT_WEIGHTS = next((p for p in [ROOT / "models" / "paper_seg.pt",
                                    ROOT / "runs" / "paper_seg" / "weights" / "best.pt"] if p.exists()),
                       ROOT / "models" / "paper_seg.pt")


# ----------------------------------------------------------------------------
# Geometry: outline -> 4 corners -> flattened image
# ----------------------------------------------------------------------------
def order_corners(pts):
    """Order 4 points clockwise starting at the top-left one."""
    pts = np.asarray(pts, np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coords (y points down)
    start = np.argmin(pts.sum(1))  # top-left has smallest x+y
    return np.roll(pts, -start, 0)


def fit_quad(poly):
    """Return (corners[4,2], is_quad). is_quad=False -> used the min-area rectangle."""
    hull = cv2.convexHull(poly.astype(np.float32))
    peri = cv2.arcLength(hull, True)
    hull_area = cv2.contourArea(hull)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) <= 4:
            break
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
        if cv2.contourArea(quad) > 0.85 * hull_area:  # quad really describes the shape
            return order_corners(quad), True
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull))), False


def rectify(frame, quad, poly=None, max_side=1600):
    """Perspective-warp the quad to a flat rectangle. If `poly` is given (non-rectangular
    paper), pixels outside the real outline are painted white."""
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 10 or h < 10:
        return None
    s = min(1.0, max_side / max(w, h))
    w, h = int(w * s), int(h * s)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    flat = cv2.warpPerspective(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    if poly is not None:
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        m = cv2.warpPerspective(m, M, (w, h), flags=cv2.INTER_NEAREST)
        flat[m == 0] = 255
    return flat


def enhance(flat):
    """Grey + local contrast (CLAHE) + sharpening: makes ink stand out for reading/OCR."""
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    return cv2.addWeighted(g, 1.5, blur, -0.5, 0)


def quality(flat, quad_area):
    """Higher = better capture. Sharpness (variance of the Laplacian, i.e. how much
    fine edge detail there is) x size in pixels x penalty for blown-out glare."""
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (int(480 * g.shape[1] / g.shape[0]), 480))
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    glare = float((g >= 250).mean())
    return sharp * np.sqrt(quad_area / 1e5) * (1 - min(0.9, glare * 3)), sharp, glare


def touches_border(poly, shape, margin=4):
    h, w = shape[:2]
    return (poly[:, 0].min() < margin or poly[:, 1].min() < margin or
            poly[:, 0].max() > w - margin or poly[:, 1].max() > h - margin)


# ----------------------------------------------------------------------------
# Per-sheet tracking state
# ----------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    cls: int
    frames: int = 0
    quad: np.ndarray = None
    best: float = 0.0
    last_save: float = 0.0
    saved: bool = False
    history: list = field(default_factory=list)

    def smooth(self, quad, frame_diag, alpha=0.5):
        """Exponential moving average of the corners; reset on big jumps."""
        if self.quad is not None and np.abs(quad - self.quad).mean() < 0.05 * frame_diag:
            self.quad = alpha * quad + (1 - alpha) * self.quad
        else:
            self.quad = quad
        return self.quad


class Scanner:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.weights)
        self.classes = [0, 1] if args.labels else [0]
        self.tracks = {}
        self.root = self.out = ROOT / "captures" / "scans" / time.strftime("%Y%m%d_%H%M%S")
        self.last_capture = None  # thumbnail shown in the corner
        self.n_forced = 0
        self.live = None
        self.last_sheet_read = 0.0
        if getattr(args, "read", False):
            from medscan.live import LiveReader
            self.live = LiveReader(self.root, min_views=args.min_views)

    def save(self, name, flat):
        self.out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.out / f"{name}.jpg"), flat, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.out / f"{name}_enhanced.jpg"), enhance(flat))
        self.last_capture = flat

    def process(self, frame, track=True, force=False):
        a = self.args
        kw = dict(conf=a.conf, imgsz=a.imgsz, classes=self.classes, retina_masks=True, verbose=False)
        r = (self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if track
             else self.model.predict(frame, **kw))[0]
        vis = frame.copy()
        if r.masks is None:
            self._fallback(frame)
            return vis, []
        diag = float(np.hypot(*frame.shape[:2]))
        frame_area = frame.shape[0] * frame.shape[1]
        found = []
        for i, poly in enumerate(r.masks.xy):
            if len(poly) < 3:
                continue
            cls = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            tid = int(r.boxes.id[i]) if r.boxes.id is not None else i
            t = self.tracks.setdefault(tid, Track(tid, cls))
            t.frames += 1

            raw_quad, is_quad = fit_quad(poly)
            # smoothed corners for display (less jitter); exact corners of THIS
            # frame for the capture (smoothing lags behind a moving sheet)
            quad = t.smooth(raw_quad, diag) if track else raw_quad
            area = cv2.contourArea(raw_quad.astype(np.float32))
            name = f"{CLASS_NAMES[cls]}_id{tid}"

            # --- capture decision
            ok = (t.frames >= a.min_frames or not track) and area > a.min_area * frame_area \
                and not touches_border(poly, frame.shape)
            if ok or force:
                flat = rectify(frame, raw_quad, None if is_quad else poly)
                if flat is not None:
                    if self.live and ok:
                        if self.live.offer(name, flat, raw_quad, frame.shape):
                            self.last_sheet_read = time.time()
                    score, sharp, glare = quality(flat, area)
                    now = time.time()
                    if force:
                        self.n_forced += 1
                        self.save(f"{name}_manual{self.n_forced}", flat)
                    elif score > t.best * 1.1 and now - t.last_save > 0.3:
                        t.best, t.last_save, t.saved = score, now, True
                        self.save(name, flat)
                        print(f"captured {name}: sharpness {sharp:.0f}, glare {glare:.1%}, score {score:.0f}")
            found.append(name)

            # --- drawing
            col = COLORS[cls]
            cv2.polylines(vis, [poly.astype(np.int32)], True, col, 1)
            cv2.polylines(vis, [quad.astype(np.int32)], True, col, 3)
            for p in quad.astype(np.int32):
                cv2.circle(vis, tuple(p), 6, (0, 0, 255), -1)
            txt = f"{CLASS_NAMES[cls]} #{tid} {conf:.2f}" + ("  [captured]" if t.saved else "")
            if self.live:
                txt += f"  {self.live.status(name)}"
            x, y = quad[0].astype(int)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x, y - th - 10), (x + tw + 6, y), col, -1)
            cv2.putText(vis, txt, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        self._fallback(frame)
        return vis, found

    def _fallback(self, frame, idle=1.5):
        """Read the whole frame when no outlined sheet has been read for a while: covers
        documents the detector misses (e.g. glossy cards) even if it outlines something else."""
        if self.live and time.time() - self.last_sheet_read > idle:
            self.live.offer_frame(frame)

    def overlay_thumb(self, vis):
        if self.last_capture is None:
            return vis
        th = self.last_capture
        s = min(220 / th.shape[1], 220 / th.shape[0])
        th = cv2.resize(th, (max(1, int(th.shape[1] * s)), max(1, int(th.shape[0] * s))))
        h, w = th.shape[:2]
        H, W = vis.shape[:2]
        if h + 20 < H and w + 20 < W:
            vis[10:10 + h, W - w - 10:W - 10] = th
            cv2.rectangle(vis, (W - w - 11, 9), (W - 9, 11 + h), (255, 255, 255), 2)
            cv2.putText(vis, "last capture", (W - w - 10, 28 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
def run_stream(sc, args):
    src = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, int):  # ask the webcam for HD (more pixels on the text); fall back if black
        cap, note = open_camera(src, args.cam_width, args.cam_height)
        if cap is None:
            raise SystemExit(f"No camera gave a picture ({note}). Check:\n{camera_advice()}")
        print(note)
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source: {args.source}")
    writer, fps, t_prev, force = None, 0.0, time.time(), False
    black_since, warned = None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if isinstance(src, int) and is_black(frame):  # the camera went dark while running
            black_since = black_since or time.time()
            if time.time() - black_since > 3 and not warned:
                print(f"The camera picture is black. Check:\n{camera_advice()}")
                warned = True
        else:
            black_since, warned = None, False
        vis, _ = sc.process(frame, track=True, force=force)
        if black_since and time.time() - black_since > 3:
            cv2.putText(vis, "Camera picture is black: see the terminal", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        force = False
        vis = sc.overlay_thumb(vis)
        if sc.live:
            if args.auto_accept and sc.live.current():  # testing without a keyboard
                sc.live.accept()
            vis = sc.live.draw_bar(vis)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(vis, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if args.save_video:
            if writer is None:
                sc.out.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(sc.out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 30, vis.shape[1::-1])
            writer.write(vis)
        if args.show:
            cv2.imshow("scan", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                force = True
            if sc.live and key in (13, 10):  # Enter
                sc.live.accept()
            if sc.live and key in (8, 127, ord("r")):  # Backspace (Windows / Mac) or r
                sc.live.reject()
            if key == ord("s"):
                d = ROOT / "captures"
                d.mkdir(exist_ok=True)
                p = d / f"cap_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
                cv2.imwrite(str(p), frame)
                print(f"saved raw frame {p}")
    cap.release()
    if writer:
        writer.release()


def run_images(sc, paths, args):
    base = sc.root
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sc.tracks.clear()
        sc.out = base / p.stem if len(paths) > 1 else base  # one sub-folder per image
        vis, found = sc.process(img, track=False)
        print(f"{p.name}: {found or 'nothing found'}")
        if args.save_video:  # for images: save the annotated picture
            sc.out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sc.out / "annotated.jpg"), vis)
        if args.show:
            cv2.imshow("scan", sc.overlay_thumb(vis))
            if cv2.waitKey(0 if len(paths) == 1 else 700) & 0xFF == ord("q"):
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--source", default="0", help="webcam index, video file, image, or folder")
    ap.add_argument("--labels", action="store_true", help="also track labels on bottles/boxes")
    ap.add_argument("--conf", type=float, default=0.5, help="minimum confidence to keep an object")
    ap.add_argument("--imgsz", type=int, default=416, help="inference size; use the training size")
    ap.add_argument("--min-frames", type=int, default=5, help="frames a sheet must be tracked before capture")
    ap.add_argument("--min-area", type=float, default=0.02, help="min sheet area as a fraction of the frame")
    ap.add_argument("--cam-width", type=int, default=1920, help="asked of the webcam; it falls back if unsupported")
    ap.add_argument("--cam-height", type=int, default=1080)
    ap.add_argument("--save-video", action="store_true", help="also save the annotated video/image")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--read", action="store_true", help="read name / RAMQ / DOB and propose the patient record")
    ap.add_argument("--min-views", type=int, default=3, help="distinct views needed per field before proposing")
    ap.add_argument("--auto-accept", action="store_true", help="testing: accept every proposal automatically")
    args = ap.parse_args()

    sc = Scanner(args)
    src = Path(args.source)
    if src.is_dir():
        run_images(sc, sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS), args)
    elif src.suffix.lower() in IMG_EXTS:
        run_images(sc, [src], args)
    else:
        run_stream(sc, args)
    cv2.destroyAllWindows()
    if sc.root.exists():
        print(f"\nCaptures saved in {sc.root}")


if __name__ == "__main__":
    main()
