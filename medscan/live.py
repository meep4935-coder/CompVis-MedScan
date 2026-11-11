"""
Live reading for scan.py: OCR + field extraction + voting in a background thread.

The camera loop runs at ~25 FPS but reading one page takes ~0.3-0.7 s on a laptop CPU,
so the loop never waits for it. It offers a flattened page whenever a tracked sheet shows
a NEW view (moved or sharper/blurrier: medscan.vote.is_new_view); if the reader is busy
the view is skipped. Each sheet has its own Voter; when one is confident it raises a
proposal that the physician accepts or rejects with one key.

If the paper detector sees nothing (it misses some covers and labels), the whole frame
is read about once a second as the "frame" sheet, so a detector miss doesn't block reading.
"""

import json
import queue
import threading
import time
from datetime import date

import cv2

from medscan.extract import extract
from medscan.ocr import Reader
from medscan.vote import Voter, is_new_view


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (int(480 * g.shape[1] / g.shape[0]), 480))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class LiveReader:
    def __init__(self, out_dir, min_views=3, verbose=True):
        self.reader = Reader()
        self.out_dir = out_dir
        self.min_views = min_views
        self.verbose = verbose
        self.voters = {}       # sheet id -> Voter
        self.last_view = {}    # sheet id -> (normalised quad, sharpness)
        self.pending = []      # sheet ids with a proposal waiting for accept / reject
        self.accepted = []
        self.jobs = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.busy = False
        self.last_frame_read = 0.0
        self.n_reads = 0
        self.text_px = {}      # sheet id -> typical small-text height in camera pixels
        self.min_text_px = 12
        threading.Thread(target=self._work, daemon=True).start()

    # ---------------------------------------------------------- feeding views
    def offer(self, sid, flat, quad=None, frame_shape=None):
        """Called by the camera loop for a sheet that is fully in view. Returns True if queued."""
        if self.busy or self.jobs.full():
            return False
        with self.lock:
            v = self.voters.get(sid)
            if v and v.status in ("proposed", "accepted"):
                return False
        scale = 1.0  # camera pixels per pixel of the flattened page
        if quad is not None and frame_shape is not None:
            h, w = frame_shape[:2]
            cur = ([(float(x) / w, float(y) / h) for x, y in quad], sharpness(flat))
            if not is_new_view(self.last_view.get(sid), cur):
                return False
            self.last_view[sid] = cur
            q = [tuple(map(float, p)) for p in quad]  # tl, tr, br, bl
            side = (_dist(q[0], q[3]) + _dist(q[1], q[2])) / 2
            scale = side / max(1, flat.shape[0])
        try:
            self.jobs.put_nowait((sid, flat.copy(), scale))
            return True
        except queue.Full:
            return False

    def offer_frame(self, frame, every=1.0):
        """Whole-frame fallback when no sheet is detected."""
        now = time.time()
        if now - self.last_frame_read < every:
            return False
        if self.offer("frame", frame):
            self.last_frame_read = now
            return True
        return False

    def _work(self):
        while True:
            sid, img, scale = self.jobs.get()
            self.busy = True
            try:
                up = 1.0
                if max(img.shape[:2]) < 900:  # small sheet in view: upscale so text is ~30 px tall
                    up = 900 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
                items = self.reader(img)
                if items:  # how tall is the text in the CAMERA image? (< ~12 px reads poorly)
                    hs = sorted(it[0][3] - it[0][1] for it in items)
                    self.text_px[sid] = hs[len(hs) // 4] / up * scale  # smaller text (lower quartile)
                reading = extract(items)
                with self.lock:
                    v = self.voters.setdefault(sid, Voter(min_views=self.min_views))
                    rec = v.add(reading)
                    self.n_reads += 1
                    if rec and sid not in self.pending:
                        self.pending.append(sid)
                if self.verbose:
                    d = reading.as_dict()
                    print(f"[read {sid}] {d['last']}, {d['first']} | {d['ramq']} | {d['dob']} "
                          f"| agree name={d['agree_name']} dob={d['agree_dob']} | text {self.text_px.get(sid, 0):.0f}px")
            except Exception as e:  # never kill the camera loop
                print(f"[read {sid}] error: {e}")
            finally:
                self.busy = False

    # ---------------------------------------------------------- one-tap confirm
    def current(self):
        """(sheet id, proposal) waiting for the physician, or None."""
        with self.lock:
            while self.pending:
                sid = self.pending[0]
                v = self.voters.get(sid)
                if v and v.proposal:
                    return sid, v.proposal
                self.pending.pop(0)
        return None

    def accept(self):
        cur = self.current()
        if not cur:
            return None
        sid, _ = cur
        with self.lock:
            rec = self.voters[sid].accept()
            self.pending.pop(0)
        rec = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in rec.items()}
        self.accepted.append(rec)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / f"patient_{len(self.accepted)}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ACCEPTED -> {p}")
        return rec

    def reject(self):
        cur = self.current()
        if not cur:
            return None
        sid, _ = cur
        with self.lock:
            status = self.voters[sid].reject()
            self.pending.pop(0)
            self.last_view.pop(sid, None)
        print(f"rejected ({status})")
        return status

    def status(self, sid):
        with self.lock:
            v = self.voters.get(sid)
            if not v:
                return "reading"
            if v.status == "reading":
                if self.text_px.get(sid, 99) < self.min_text_px:
                    return "text too small: bring closer"
                return f"reading ({v.views} views)"
            return {"need_better_view": "closer / more light please"}.get(v.status, v.status)

    # ---------------------------------------------------------- drawing
    def draw_bar(self, vis):
        cur = self.current()
        H, W = vis.shape[:2]
        if "frame" in self.voters and time.time() - self.last_frame_read < 2:
            cv2.putText(vis, f"whole view: {self.status('frame')}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if not cur:
            if self.accepted:
                a = self.accepted[-1]
                txt = f"Accepted: {a['last'].upper()}, {a['first']}  NAM {a['ramq']}  DDN {a['dob']}"
                _bar(vis, txt, (40, 120, 40))
            return vis
        _, p = cur
        middle = f" {p['middle']}" if p.get("middle") else ""
        r = p["ramq"]
        txt = (f"{p['last'].upper()}, {p['first']}{middle}   NAM {r[:4]} {r[4:8]} {r[8:]}   "
               f"DDN {p['dob'].isoformat() if isinstance(p['dob'], date) else p['dob']}"
               f"{' (from NAM)' if p.get('dob_source') == 'ramq' else ''}   {p.get('sex', '')}")
        _bar(vis, txt, (150, 90, 0), "Enter = accept   Backspace / r = reject")
        return vis


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _bar(vis, text, color, hint=None):
    H, W = vis.shape[:2]
    h = 70 if hint else 44
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, H - h), (W, H), color, -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.putText(vis, _ascii(text), (12, H - h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    if hint:
        cv2.putText(vis, hint, (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)


def _ascii(s):
    """OpenCV's built-in font has no accents: 'Marie-Ève' -> 'Marie-Eve' (display only;
    the saved record keeps the accents)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
