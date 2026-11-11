import unittest

import numpy as np

from medscan.camera import is_black, open_camera


class FakeCam:
    """Camera 0 is black when asked for 1920x1080 (or always, if dead); camera 1 works."""
    def __init__(self, index, dead=False):
        self.index, self.dead, self.size = index, dead, None

    def isOpened(self):
        return self.index in (0, 1)

    def set(self, prop, value):
        self.size = value if self.size is None else self.size

    def read(self):
        black = self.index == 0 and (self.dead or self.size == 1920)
        f = np.zeros((720, 1280, 3), np.uint8)
        if not black:
            f[:] = np.random.default_rng(0).integers(40, 200, f.shape, dtype=np.uint8)
        return True, f

    def release(self):
        pass


class TestCamera(unittest.TestCase):
    def test_is_black(self):
        self.assertTrue(is_black(np.zeros((10, 10, 3), np.uint8)))
        dark_room = np.random.default_rng(1).integers(0, 25, (10, 10, 3), dtype=np.uint8)
        self.assertTrue(not is_black(dark_room))

    def test_falls_back_to_cameras_own_size(self):
        cap, note = open_camera(0, 1920, 1080, wait=0.05, capture=FakeCam)
        self.assertTrue(cap.index == 0 and cap.size is None)
        self.assertTrue("black picture" in note)

    def test_falls_back_to_another_camera(self):
        cap, note = open_camera(0, 1920, 1080, wait=0.05, capture=lambda i: FakeCam(i, dead=True))
        self.assertTrue(cap.index == 1 and note.startswith("camera 1"))

    def test_reports_when_nothing_works(self):
        cap, note = open_camera(0, 1920, 1080, wait=0.05, capture=lambda i: FakeCam(i if i == 0 else 9, dead=True))
        self.assertTrue(cap is None and "camera 0" in note)
