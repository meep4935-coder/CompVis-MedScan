import unittest

from medscan import ctc
from medscan.extract import extract

BLANK = [["", 0.99]]


def step(c, p=0.98):
    return [[c, p]]


def frames_for(text, overrides=None):
    """Fake recogniser output: each char on one step followed by a blank step."""
    out = []
    for i, c in enumerate(text):
        out.append(overrides.get(i, step(c)) if overrides else step(c))
        out.append(BLANK)
    return out


class TestCTC(unittest.TestCase):
    def test_letter_position_prefers_letter(self):
        fr = frames_for("GIR335081718", {3: [["3", 0.5], ["J", 0.4], ["", 0.05]]})
        self.assertEqual(ctc.decode(fr, ctc.RAMQ)[0], "GIRJ35081718")

    def test_missing_digit_recovered(self):
        fr = frames_for("GIRJ3508171") + [[["", 0.6], ["8", 0.35]], BLANK]
        self.assertEqual(ctc.decode(fr, ctc.RAMQ)[0], "GIRJ35081718")

    def test_doubled_char_needs_blank(self):
        # '00' must come from two separate runs of '0'
        self.assertEqual(ctc.decode(frames_for("1985-05-00"[:8] + "01"), ctc.DATE_YMD)[0], "1985-05-01")

    def test_greedy_spans(self):
        sp = ctc.spans(frames_for("AB"))
        self.assertEqual([s[0] for s in sp], ["A", "B"])

    def test_extract_uses_probabilities(self):
        text = "NAM: GIR3 3508 1718"
        fr = frames_for(text, {8: [["3", 0.5], ["J", 0.4], ["", 0.05]]})
        items = [([20, 0, 600, 30], "GIRARD, Jean Robert", 0.95, frames_for("GIRARD, Jean Robert")),
                 ([20, 40, 600, 70], text, 0.95, fr),
                 ([20, 80, 600, 110], "D.N.: 1935-08-17", 0.95, frames_for("D.N.: 1935-08-17"))]
        r = extract(items)
        self.assertEqual(r.ramq.value, "GIRJ35081718")
        self.assertTrue(r.agree_name and r.agree_dob)


class TestNameEvidence(unittest.TestCase):
    def test_letter_spaced_header_merged_with_typed_line(self):
        # block capitals with wide letter spacing come back in pieces ('SÉBAS TIEN')
        lines = ["Nom: GAGNON", "Prénom: SÉBAS TIEN", "NAM: GAGS 9011 2188", "Patient: GAGNON, Sébastien"]
        items = [([20, 40 * i, 600, 40 * i + 30], t, 0.95, frames_for(t)) for i, t in enumerate(lines)]
        r = extract(items)
        self.assertEqual((r.first.value, r.middle.value, r.last.value), ("Sébastien", "", "Gagnon"))

    def test_least_certain_letter_decides(self):
        # 'Peme' (rr read as m, with low confidence) vs a clean 'PIERRE' in the ID block
        typed = "Patient: TURGEON, Peme"
        unsure = {typed.index("m"): [["m", 0.35], ["r", 0.3], ["", 0.2]]}
        lines = [("Nom: TURGEON", None), ("Prénom: PIERRE", None), ("NAM: TURP 1505 1855", None), (typed, unsure)]
        items = [([20, 40 * i, 600, 40 * i + 30], t, 0.95, frames_for(t, o)) for i, (t, o) in enumerate(lines)]
        self.assertEqual(extract(items).first.value, "Pierre")


if __name__ == "__main__":
    unittest.main()
