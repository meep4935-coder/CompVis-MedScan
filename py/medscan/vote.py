"""
Keep reading until confident, then propose; on reject, keep reading.

One Voter per tracked document (or per patient once documents are linked). Feed it the
Reading of every *new view* of that document (see is_new_view: 20 frames of the same blurry
view are not 20 independent votes). Each field is voted on separately, weighted by its
confidence. A record is proposed only when:
  - every required field has enough votes from distinct views and a clear majority, and
  - the winners agree with each other (RAMQ letters = name, RAMQ digits = DOB).

The physician answers with one tap:
  accept()  -> done
  reject()  -> that exact record is never proposed again; votes are cleared so the next
               proposal comes from NEW views, not from the runner-up of the same evidence.
               After `max_rejects` rejections the voter asks for a better view instead.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from medscan.extract import confident_still
from medscan.ramq import decode, name_code

FIELDS = ("last", "first", "middle", "dob", "ramq")


def is_new_view(prev, cur, move=0.04, sharp=1.25):
    """prev/cur: (quad corners as 4 (x, y) pairs normalised to 0-1, sharpness score).
    A view counts as new when the page moved, or got noticeably sharper/blurrier."""
    if prev is None:
        return True
    (q0, s0), (q1, s1) = prev, cur
    shift = max(abs(a - b) for p0, p1 in zip(q0, q1) for a, b in zip(p0, p1))
    ratio = max(s0, s1) / max(1e-6, min(s0, s1))
    return shift > move or ratio > sharp


@dataclass
class Voter:
    min_views: int = 3          # distinct views needed per field
    min_share: float = 0.6      # winner's share of the (confidence-weighted) votes
    min_conf: float = 0.3       # ignore field reads below this confidence
    max_rejects: int = 2
    require: tuple = ("last", "first", "dob", "ramq")
    votes: dict = field(default_factory=lambda: {f: defaultdict(float) for f in FIELDS})
    counts: dict = field(default_factory=lambda: {f: defaultdict(int) for f in FIELDS})
    shown: dict = field(default_factory=lambda: {f: {} for f in FIELDS})
    dob_read: set = field(default_factory=set)  # DOB values seen printed (vs. worked out from RAMQ)
    require_sure: bool = False  # tried: in simulated live sessions it cost 2 right and fixed no wrong
    sure: set = field(default_factory=set)      # (name, RAMQ) that some single view was sure of
    rejected: list = field(default_factory=list)
    views: int = 0
    proposal: dict = None
    accepted: dict = None

    def add(self, reading):
        """Add one new view's Reading. Returns a proposal dict when ready, else None."""
        if self.accepted or self.proposal:
            return None
        self.views += 1
        for f in FIELDS:
            fv = getattr(reading, f)
            if f == "middle" and reading.first.value is not None:
                val = fv.value or ""  # 'no middle name' is a real answer
            else:
                val = fv.value
            if val is None or (fv.conf < self.min_conf and val != ""):
                continue
            key = val.isoformat() if isinstance(val, date) else val.casefold()
            self.votes[f][key] += max(fv.conf, 0.3)
            self.counts[f][key] += 1
            self.shown[f].setdefault(key, val)  # how to display it ('Tremblay', not 'tremblay')
            if f == "dob" and getattr(reading, "dob_source", "read") == "read":
                self.dob_read.add(key)
        if confident_still(reading):  # this single view alone was sure of its name and RAMQ
            self.sure.add(self._who(reading.last.value, reading.first.value, reading.middle.value,
                                    reading.ramq.value))
        return self._maybe_propose()

    @staticmethod
    def _who(last, first, middle, ramq):
        return ((last or "").casefold(), " ".join(x for x in (first, middle) if x).casefold(), ramq)

    def _winner(self, f):
        if not self.votes[f]:
            return None, 0, 0.0
        total = sum(self.votes[f].values())
        key, w = max(self.votes[f].items(), key=lambda kv: kv[1])
        return self.shown[f][key], self.counts[f][key], w / total

    def _maybe_propose(self):
        rec = {}
        for f in FIELDS:
            val, n, share = self._winner(f)
            if f in self.require and (val is None or n < self.min_views or share < self.min_share):
                return None
            rec[f] = val
        if rec["ramq"]:
            if rec["last"] and name_code(rec["last"], rec["first"] or "") != rec["ramq"][:4]:
                return None
            d = decode(rec["ramq"])
            if rec["dob"] and (rec["dob"].year % 100, rec["dob"].month, rec["dob"].day) != (d["yy"], d["month"], d["day"]):
                return None
            rec["sex"] = d["sex"]
        # was the DOB ever actually read off the page, or only worked out from the RAMQ?
        rec["dob_source"] = "read" if rec["dob"] and rec["dob"].isoformat() in self.dob_read else "ramq"
        # a misread repeated in every view wins the vote; also ask that at least one view was
        # individually sure of exactly this name and RAMQ (medscan.extract.confident_still)
        if self.require_sure and self._who(rec["last"], rec["first"], rec["middle"], rec["ramq"]) not in self.sure:
            return None
        sig = self._sig(rec)
        if sig in self.rejected:
            return None
        self.proposal = rec
        return rec

    @staticmethod
    def _sig(rec):
        return tuple(str(rec.get(f) or "").casefold() for f in FIELDS)

    def accept(self):
        self.accepted, self.proposal = self.proposal, None
        return self.accepted

    def reject(self):
        """Returns 'reading' (keep pointing the camera) or 'need_better_view' (show a hint)."""
        if self.proposal:
            self.rejected.append(self._sig(self.proposal))
        self.proposal = None
        self.votes = {f: defaultdict(float) for f in FIELDS}
        self.counts = {f: defaultdict(int) for f in FIELDS}
        self.shown = {f: {} for f in FIELDS}
        self.dob_read = set()
        self.sure = set()
        self.views = 0
        if len(self.rejected) >= self.max_rejects:
            return "need_better_view"
        return "reading"

    @property
    def status(self):
        if self.accepted:
            return "accepted"
        if self.proposal:
            return "proposed"
        if len(self.rejected) >= self.max_rejects:
            return "need_better_view"
        return "reading"
