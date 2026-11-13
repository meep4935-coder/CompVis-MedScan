"""
Browser entry point for the reading rules.

The JavaScript reader hands over the text lines it read (with the recogniser's runner-up
characters); the SAME Python files as the desktop (medscan/ramq.py, ctc.py, extract.py,
vote.py) turn them into a patient record. Nothing here is a copy of the rules: this file
only passes data in and out as JSON.
"""

import json
from datetime import date

from medscan.extract import extract, still_ok
from medscan.vote import Voter

MIN_VIEWS = 3
voter = Voter(min_views=MIN_VIEWS)


def reset():
    """Start over with a new patient."""
    global voter
    voter = Voter(min_views=MIN_VIEWS)


def reject():
    """The physician rejected the proposal: keep reading, never propose that record again."""
    return voter.reject()


def accept():
    return _plain(voter.accept())


def _plain(rec):
    return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in (rec or {}).items()}


def read(items_json, live):
    """items: [[x0, y0, x1, y1], text, confidence, frames] per text line.
    live=True: one more camera view, voted with the earlier ones.
    live=False: a single photo, accepted only if it is sure on its own."""
    items = [(b, t, c, f) for b, t, c, f in json.loads(items_json)]
    rd = extract(items)
    sure, why = still_ok(rd)
    out = {"reading": rd.as_dict(), "sure": sure, "why": why}
    if live:
        voter.add(rd)
        out["views"] = voter.views
        out["status"] = voter.status
        out["proposal"] = _plain(voter.proposal) if voter.proposal else None
    return json.dumps(out, ensure_ascii=False)
