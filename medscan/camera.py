"""
Open a webcam and make sure it actually gives a picture.

Some cameras open fine but send only black frames: the size asked for isn't supported,
another app (or an earlier scan window) holds the camera, the laptop lid is closed, the
terminal has no camera permission, or index 0 is a phone (Continuity Camera on a Mac).
open_camera() tries the asked size, then the camera's own size, then the other cameras,
and says what it found instead of showing a black window.
"""

import sys
import time

import cv2


def is_black(frame):
    """A dead feed: (almost) every pixel black. A dark room still has some light and noise."""
    if frame is None:
        return True
    f = frame[::8, ::8]  # every 8th pixel is plenty, and cheap enough to check every frame
    return float(f.mean()) < 3 and float(f.std()) < 3


def advice():
    if sys.platform == "darwin":
        return ("  - Quit other apps using the camera (FaceTime, Zoom, Teams, Photo Booth, a browser\n"
                "    tab) and any earlier scan window (press q in it, or close its Terminal window).\n"
                "  - Open the laptop lid: the built-in camera is off when it is closed.\n"
                "  - System Settings > Privacy & Security > Camera: turn on Terminal (or the app you\n"
                "    run this from), then quit and reopen that app.\n"
                "  - If an iPhone is nearby it may be used as the camera: try  --source 1")
    return ("  - Close other apps using the camera, and any earlier scan window (press q in it).\n"
            "  - Check the camera's privacy shutter or cover.\n"
            "  - Windows Settings > Privacy > Camera: allow desktop apps to use the camera.\n"
            "  - Try another camera:  --source 1")


def open_camera(index, width=None, height=None, wait=3.0, capture=cv2.VideoCapture, others=4):
    """Returns (cap, note) for the first camera that gives a picture, or (None, report)."""
    tries = []
    if width and height:
        tries.append((index, (width, height)))
    tries.append((index, None))
    tries += [(i, None) for i in range(others) if i != index]
    report, missing = [], set()
    for i, size in tries:
        if i in missing:
            continue
        cap = capture(i)
        if not cap.isOpened():
            cap.release()
            missing.add(i)
            if i == index:
                report.append(f"camera {i}: could not be opened")
            continue
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        got, t0 = None, time.time()
        while time.time() - t0 < wait:  # the first frames are often black while it starts
            ok, f = cap.read()
            if ok and not is_black(f):
                got = f
                break
        what = f"camera {i}" + (f" asked for {size[0]}x{size[1]}" if size else "")
        if got is not None:
            note = f"camera {i}: {got.shape[1]}x{got.shape[0]}"
            if report:
                note += "  (" + "; ".join(report) + ")"
            return cap, note
        cap.release()
        report.append(f"{what}: black picture")
    return None, "; ".join(report) or f"camera {index}: could not be opened"
