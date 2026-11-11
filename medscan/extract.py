"""
Pull the patient's name, RAMQ number and date of birth out of OCR results, with rules
instead of human correction.

Input: OCR items for one page or frame, each (bbox [x0, y0, x1, y1], text, confidence).
Output: a Reading with the best name / RAMQ / DOB and a confidence for each.

How uncertainty is resolved, in order:
  1. Field rules: a name has no digits and its words start with capitals; a DOB is a real,
     past date in a Quebec format; a RAMQ is 4 letters + 8 digits with a valid date.
  2. Labels: "NAM", "DDN", "Né(e)", "Patient :"... make a candidate more likely, while
     "Date :", "Exp", "Dr" make it less likely (visit date, expiry, physician).
  3. Cross-checks: the RAMQ letters must match the family name + first initial; its digits
     must match the DOB (+50 on the month for women). The combination where all three agree
     wins, and a RAMQ letter that differs from the name only by a look-alike (I/J/1/3)
     is repaired from the name.
This module deliberately does NOT use the generator's name lists: it must work on real names.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from itertools import product

from medscan import ctc
from medscan.ramq import decode, find_ramq_loose, is_valid, name_code

TODAY = date.today()


# ---------------------------------------------------------------- re-reading under rules

_C = r"[A-Za-z0-9|/²]"
RAMQ_SPAN = re.compile(rf"(?=({_C}{{3,5}}[\s\-]?{_C}{{3,5}}[\s\-]?{_C}{{2,5}}))")
_D = r"[0-9OoQDIlLZSBG|]"
DATE_SPAN = re.compile(rf"(?=({_D}{{1,4}}\s?[-/.]\s?{_D}{{1,3}}\s?[-/.]\s?{_D}{{2,4}}))")


def refine(items):
    """For OCR items that carry per-step probabilities (medscan.ocr.Reader), re-read every
    RAMQ-like and date-like part of the line under its pattern (medscan.ctc) and put the
    result back in the text. Returns (items as (bbox, text, conf), ramq_paths) where
    ramq_paths[ramq] = (frames, t0, t1, log-prob) lets the name cross-check ask
    'how likely is it that these 4 letters were really the name's code?'."""
    out, paths = [], {}
    for it in items:
        b, t, c = it[:3]
        frames = it[3] if len(it) > 3 else None
        if not frames:
            out.append((b, t, c))
            continue
        sp = ctc.spans(frames)
        if "".join(s[0] for s in sp) != t:
            out.append((b, t, c))
            continue
        edits = []  # (score, start, end, new_text)
        for m in RAMQ_SPAN.finditer(t):
            g, i, j = m[1], m.start(1), m.end(1)
            if sum(ch.isalpha() for ch in g[:5]) < 2 or sum(ch.isdigit() for ch in g) < 4:
                continue
            t0, t1 = ctc.window(frames, sp, i, j - 1)
            txt, total, n = ctc.decode(frames, ctc.RAMQ, t0, t1)
            if txt:
                r = re.sub(r"[\s\-]", "", txt).upper()
                if is_valid(r) and total / max(1, n) > -0.8:
                    edits.append((total / n, i, j, txt.upper()))
                    paths[r] = (frames, t0, t1, total)
        for m in DATE_SPAN.finditer(t):
            g, i, j = m[1], m.start(1), m.end(1)
            if sum(ch.isdigit() for ch in g) < 4:
                continue
            t0, t1 = ctc.window(frames, sp, i, j - 1)
            best = None
            for pat in (ctc.DATE_YMD, ctc.DATE_DMY):
                txt, total, n = ctc.decode(frames, pat, t0, t1)
                if txt and parse_dates(txt) and (best is None or total > best[0]):
                    best = (total, n, txt)
            if best and best[0] / max(1, best[1]) > -0.8:
                edits.append((best[0] / best[1], i, j, best[2]))
        # apply the best non-overlapping edits, right to left
        chosen = []
        for e in sorted(edits, reverse=True):
            if all(e[2] <= o[1] or e[1] >= o[2] for o in chosen):
                chosen.append(e)
        for _, i, j, new in sorted(chosen, key=lambda e: -e[1]):
            t = t[:i] + new + t[j:]
        out.append((b, t, c))
    return out, paths


def letters_plausible(paths, ramq, code, max_drop=6.0):
    """Could the RAMQ's 4 letters really be `code`? Re-read them forced to `code` and compare
    with the best free read: a small log-prob drop means the model had `code` as a close
    second guess (e.g. J 0.40 behind 3 0.45). None = no probabilities to judge with."""
    if ramq not in paths:
        return None
    frames, t0, t1, total = paths[ramq]
    txt, forced, _ = ctc.decode(frames, ctc.ramq_fixed(code), t0, t1)
    if not txt:
        return False, None
    return total - forced <= max_drop, re.sub(r"[\s\-]", "", txt).upper()


def fold(s):
    """Upper-case, accents removed: 'Né(e)' -> 'NE(E)'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).upper()


def fuzzy_in(word, choices, cutoff=0.75):
    return bool(difflib.get_close_matches(fold(word), choices, n=1, cutoff=cutoff))


# ---------------------------------------------------------------- rows

def rows(items):
    """Merge OCR boxes that sit side by side on one text line (EasyOCR returns words).
    Large horizontal gaps keep separate columns apart."""
    items = sorted(items, key=lambda o: (o[0][1] + o[0][3]) / 2)
    used = [False] * len(items)
    out = []
    for i, (b, t, c) in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        group = [(b, t, c)]
        h = b[3] - b[1]
        cy = (b[1] + b[3]) / 2
        changed = True
        while changed:
            changed = False
            for j, (b2, t2, c2) in enumerate(items):
                h2 = b2[3] - b2[1]
                if used[j] or abs((b2[1] + b2[3]) / 2 - cy) > 0.5 * h or max(h, h2) > 1.4 * min(h, h2):
                    continue  # other line, or different text size (a header next to a sticker)
                x0, x1 = min(g[0][0] for g in group), max(g[0][2] for g in group)
                gap = max(b2[0] - x1, x0 - b2[2])
                if gap < 0.8 * h:  # about a word space; more than that = another column
                    group.append((b2, t2, c2))
                    used[j] = changed = True
        group.sort(key=lambda g: g[0][0])
        box = [min(g[0][0] for g in group), min(g[0][1] for g in group),
               max(g[0][2] for g in group), max(g[0][3] for g in group)]
        conf = sum(g[2] for g in group) / len(group)
        out.append((box, " ".join(g[1] for g in group), conf))
    return out


def above(rs, r):
    """The row just above r and overlapping it horizontally (for label-above-value layouts)."""
    best = None
    for o in rs:
        if o is r or o[0][3] > r[0][1] + 5 or o[0][2] < r[0][0] or o[0][0] > r[0][2]:
            continue
        if best is None or o[0][3] > best[0][3]:
            best = o
    if best and r[0][1] - best[0][3] < 1.5 * (r[0][3] - r[0][1]):
        return best
    return None


# ---------------------------------------------------------------- candidates

@dataclass
class Cand:
    value: object
    score: float
    text: str
    conf: float
    bbox: list = field(default_factory=list)


RAMQ_LABELS = ["NAM", "RAMQ", "NO RAMQ", "ASS. MALADIE", "NO ASS. MALADIE", "ASSURANCE MALADIE", "HIN"]


def ramq_candidates(rs):
    out = []
    for i, (b, t, c) in enumerate(rs):
        for r in find_ramq_loose(t):
            s = 1.0 + c
            if re.search(r"\b(NAM|RAMQ|RAMO|MALADIE|ASS)\b", fold(t)):
                s += 1.0
            out.append(Cand(r, s, t, c, b))
    return out


MONTHS = {"JANVIER": 1, "JAN": 1, "FEVRIER": 2, "FEV": 2, "MARS": 3, "MAR": 3, "AVRIL": 4, "AVR": 4,
          "MAI": 5, "JUIN": 6, "JUN": 6, "JUILLET": 7, "JUIL": 7, "JUL": 7, "AOUT": 8, "AOU": 8,
          "SEPTEMBRE": 9, "SEPT": 9, "SEP": 9, "OCTOBRE": 10, "OCT": 10, "NOVEMBRE": 11, "NOV": 11,
          "DECEMBRE": 12, "DEC": 12,
          "JANUARY": 1, "FEBRUARY": 2, "FEB": 2, "MARCH": 3, "APRIL": 4, "APR": 4, "MAY": 5, "JUNE": 6,
          "JULY": 7, "AUGUST": 8, "AUG": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12}
DIGITISH = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "!": "1", "Z": "2", "S": "5",
            "B": "8", "G": "6", "T": "7"}
DOB_LABEL = re.compile(r"(?<![A-Z])(DDN|D\. ?N\.|NE\(E\)|NEE?|NAISSANCE|DOB|BIRTH)(?![A-Z])")
OTHER_DATE_LABEL = re.compile(r"(?<![A-Z])(DATE|EXP|EXPIRATION|HEURE|VISITE|RX|SERVI|DELIVRE|RENOUV)(?![A-Z])")


def _digits_fixed(s):
    """In chunks that are mostly digits, turn look-alike letters into digits: '19B5-O4-l2' -> '1985-04-12'.
    Words like month names are left alone."""
    def fix(m):
        w = m.group()
        if len(w) <= 4 and sum(ch.isdigit() for ch in w) * 2 >= len(w):
            return "".join(DIGITISH.get(ch, ch) for ch in w)
        return w
    return re.sub(r"[0-9A-Z|!]+", fix, s)


def _valid_dob(y, m, d):
    try:
        x = date(y, m, d)
    except ValueError:
        return None
    if x > TODAY or TODAY.year - y > 120:
        return None
    return x


def parse_dates(text):
    """All plausible birth dates in a line, as (date, how_sure) pairs."""
    t = _digits_fixed(fold(text))
    found = []
    for m in re.finditer(r"(?<!\d)(\d{4})\s?[-/.]\s?(\d{1,2})\s?[-/.]\s?(\d{1,2})(?!\d)", t):
        x = _valid_dob(int(m[1]), int(m[2]), int(m[3]))
        if x:
            found.append((x, 1.0))
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s?[-/.]\s?(\d{1,2})\s?[-/.]\s?(\d{4})(?!\d)", t):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        x = _valid_dob(y, b, a)  # Quebec: JJ/MM/AAAA
        if x:
            found.append((x, 0.9))
        x2 = _valid_dob(y, a, b)  # MM/DD/YYYY (English forms) is possible but less likely
        if x2 and x2 != x:
            found.append((x2, 0.4))
    for m in re.finditer(r"(?<!\d)(\d{1,2})(?:ER)?\s?[-.]?\s?([A-Z]{3,9})\.?\s?[-.]?\s?(\d{4})(?!\d)", t):
        mon = difflib.get_close_matches(m[2], MONTHS, n=1, cutoff=0.7)
        if mon:
            x = _valid_dob(int(m[3]), MONTHS[mon[0]], int(m[1]))
            if x:
                found.append((x, 1.0 if mon[0] == m[2] else 0.7))
    return found


def dob_candidates(rs):
    out = []
    for r in rs:
        b, t, c = r
        ft = fold(t)
        for x, sure in parse_dates(t):
            s = sure + c * 0.5
            up = above(rs, r)
            if DOB_LABEL.search(ft) or (up and DOB_LABEL.search(fold(up[1]))):
                s += 2.0
            elif OTHER_DATE_LABEL.search(ft):
                s -= 2.0
            age = re.search(r"\(?(\d{1,3})\s*ANS\)?", ft)
            if age:
                years = TODAY.year - x.year - ((TODAY.month, TODAY.day) < (x.month, x.day))
                s += 1.0 if abs(int(age[1]) - years) <= 1 else -0.5
            if x.year >= TODAY.year - 1:  # this year's date: much more likely a visit date
                s -= 1.0
            out.append(Cand(x, s, t, c, b))
    return out


NAME_LABELS = ["NOM", "PATIENT", "PATIENTE", "NOM DU PATIENT", "NAME", "USAGER", "BENEFICIAIRE"]
GIVEN_LABELS = ["PRENOM", "PRENOMS"]
HONORIFICS = {"MME", "M", "M.", "MR", "MRS", "MS", "MLLE", "MADAME", "MONSIEUR"}
STOP = set("""HOPITAL CENTRE CHU CHUM CUSM CISSS CIUSSS INSTITUT CLINIQUE GMF PHARMACIE CARTE NOTE NOTES
EVOLUTION ORDONNANCE ORDONNANCES FEUILLE CONSULTATION URGENCE DATE HEURE ALLERGIE ALLERGIES PLAN RX DOSSIER
SEXE NAM RAMQ DDN PERMIS DR DRE DOCTEUR MEDECIN QTE REN RUE BOUL AV TEL MG MCG ML COMPRIME COMPRIMES
CAPSULE SIGNATURE TELEPHONE ADRESSE EXP SERVICE UNITE CHAMBRE LIT ETAGE LABORATOIRE RADIOLOGIE
QUEBEC MONTREAL LAVAL""".split())
TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "4": "A", "3": "E", "7": "T", "2": "Z"}


def _clean_name_tokens(text):
    """Apply the name rules to a line; returns tokens or None if this can't be a name."""
    t = text.strip().strip(".,;:")
    t = re.sub(r"^(?:M|Mme|Mlle|Mr|Mrs|Ms|Madame|Monsieur)\.?\s+", "", t, flags=re.I)  # titles
    t = re.sub(r"(?<=[^\W\d_])[.;]\s", ", ", t)  # 'DUBÉ. Marie-Ève': OCR read the comma as a period
    toks = t.replace(",", " , ").split()
    if sum(ch.isdigit() for ch in t) > 1:
        return None
    toks = ["".join(TO_LETTER.get(ch, ch) for ch in w) for w in toks]  # a digit inside a name is a misread
    while toks and fold(toks[0]).rstrip(".") in HONORIFICS:
        toks = toks[1:]
    words = [w for w in toks if w != ","]
    if not 2 <= len(words) <= 5:
        return None
    for w in words:
        core = w.strip("'")
        if not re.fullmatch(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", core):
            return None
        if not core[0].isupper():  # names are capitalised
            return None
        if fold(core) in STOP:
            return None
    return toks


def _splits(toks):
    """Every reasonable (first, middle, last) reading of a name line, with a prior score."""
    words = [w for w in toks if w != ","]
    out = []
    if "," in toks:  # 'TREMBLAY, Marie-Ève Anne'
        k = toks.index(",")
        last, given = toks[:k], [w for w in toks[k + 1:] if w != ","]
        if len(last) == 1 and given:
            out.append(((given[0], " ".join(given[1:]), last[0]), 1.0))
        return out
    all_upper = all(w.isupper() for w in words)
    first_upper = words[0].isupper() and not all_upper
    # natural order: 'Marie-Ève Anne Tremblay'
    nat = 0.3 if first_upper else (0.6 if all_upper else 0.8)
    # family name first: 'TREMBLAY Marie-Ève Anne' / 'TREMBLAY MARIE-ÈVE' (usual on labels in capitals)
    fam = 0.9 if first_upper else (0.7 if all_upper else 0.5)
    out.append(((words[0], " ".join(words[1:-1]), words[-1]), nat))
    out.append(((words[1], " ".join(words[2:]), words[0]), fam))
    return out


def order_is_ambiguous(text):
    """'GRENIER WILLIAM GUY': all capitals, no comma -> the family name could be first or last."""
    words = [w for w in text.split() if w.isalpha() or "-" in w or "'" in w]
    return "," not in text and len(words) >= 2 and all(w.isupper() for w in words)


def name_candidates(rs):
    out = []
    last_lab = given_lab = None
    for r in rs:
        b, t, c = r
        ft = fold(t)
        label_bonus = 0.0
        body = t
        m = re.match(r"\s*([^\W\d_][^\W\d_ .]*(?: [^\W\d_]+)?)\s*([:.])\s*(.*)$", t)
        if m:
            if fuzzy_in(m[1], NAME_LABELS):
                label_bonus, body = 1.5, m[3]
            elif fuzzy_in(m[1], GIVEN_LABELS):
                if m[3].strip():
                    given_lab = (m[3], c, b)
                continue
            elif m[2] == ":" and len(m[1]) <= 12 and not fuzzy_in(m[1], list(HONORIFICS)):
                continue  # 'Allergie : ...', 'Plan : ...' -> some other field ('DUBÉ. Marie' is not)
        first_word = ft.split()[0] if ft.split() else ""
        if fuzzy_in(first_word, GIVEN_LABELS, 0.8):
            if len(t.split()) > 1:  # a lone 'PRÉNOM' piece must not erase the full line
                given_lab = (t.split(None, 1)[1], c, b)
            continue
        if fuzzy_in(first_word, ["NOM"], 0.99) and len(t.split()) >= 2:
            last_lab = (t.split(None, 1)[1], c, b)
            continue
        up = above(rs, r)
        if up and fuzzy_in(fold(up[1]).strip(" :"), NAME_LABELS):
            label_bonus = 1.0
        if re.search(r"\b(DRE?|PERMIS|MEDECIN)\b", ft):
            continue  # the physician, not the patient
        toks = _clean_name_tokens(body)
        if not toks:
            continue
        for (first, middle, last), prior in _splits(toks):
            out.append(Cand((first, middle, last), prior + label_bonus + c * 0.5, t, c, b))
    if last_lab and given_lab:  # card layout: 'NOM LEMIEUX' / 'PRÉNOM GENEVIÈVE LUCIE'
        g = given_lab[0].split()
        lt = _clean_name_tokens(last_lab[0] + " X")
        if g and lt:
            out.append(Cand((g[0], " ".join(g[1:]), last_lab[0].strip()), 3.0,
                            f"{last_lab[0]} / {given_lab[0]}", min(last_lab[1], given_lab[1]), last_lab[2]))
    return out


# ---------------------------------------------------------------- joint decision

LOOKALIKE = [set("IJL1|/3"), set("O0QD"), set("S5"), set("B8"), set("Z2"), set("G6"), set("T7"), set("A4"),
             set("E3"), set("UV"), set("MN"), set("CG")]


def lookalike(a, b):
    return a == b or any(a in g and b in g for g in LOOKALIKE)


def nice_case(s):
    """'MARIE-ÈVE' -> 'Marie-Ève', "D'AMOUR" -> "D'Amour"; keeps already mixed-case words."""
    if not s or not s.isupper():
        return s
    return re.sub(r"[^\W\d_]+", lambda m: m.group().capitalize(), s.lower())


@dataclass
class Field:
    value: object = None
    conf: float = 0.0
    checks: list = field(default_factory=list)


@dataclass
class Reading:
    first: Field = field(default_factory=Field)
    middle: Field = field(default_factory=Field)
    last: Field = field(default_factory=Field)
    dob: Field = field(default_factory=Field)
    ramq: Field = field(default_factory=Field)
    sex: str = None
    agree_name: bool = False
    agree_dob: bool = False

    def as_dict(self):
        return {"first": self.first.value, "middle": self.middle.value, "last": self.last.value,
                "dob": self.dob.value.isoformat() if self.dob.value else None, "ramq": self.ramq.value,
                "sex": self.sex, "agree_name": self.agree_name, "agree_dob": self.agree_dob,
                "conf": {k: round(getattr(self, k).conf, 2) for k in ("last", "first", "dob", "ramq")}}


def _top(cands, k):
    best = {}
    for c in cands:
        if c.value not in best or c.score > best[c.value].score:
            best[c.value] = c
    return sorted(best.values(), key=lambda c: -c.score)[:k]


def extract(items):
    items, paths = refine(items)
    rs = rows(items)
    rs += [it for it in items if it not in rs]  # also each piece alone, in case a merge was wrong
    names = _top(name_candidates(rs), 8) + [None]
    ramqs = _top(ramq_candidates(rs), 4) + [None]
    dobs = _top(dob_candidates(rs), 5) + [None]

    best, best_s, memo = None, -1e9, {}
    for n, r, d in product(names, ramqs, dobs):
        s = sum(c.score for c in (n, r, d) if c is not None)
        ag_name = ag_dob = False
        fixed_r = r.value if r else None
        if n and r:
            code = name_code(n.value[2], n.value[0])
            if code == r.value[:4]:
                ag_name, s = True, s + 4
            elif len(code) == 4:
                # repair the RAMQ letters from the name, if the reader finds that plausible
                if (r.value, code) not in memo:
                    memo[(r.value, code)] = letters_plausible(paths, r.value, code)
                judged = memo[(r.value, code)]
                if judged is None:  # no probabilities (other engines): look-alike table
                    ok, cand = all(lookalike(a, b) for a, b in zip(code, r.value[:4])), code + r.value[4:]
                else:
                    ok, cand = judged
                if ok and cand and is_valid(cand):
                    ag_name, fixed_r, s = True, cand, s + 3
                else:
                    s -= 2
            else:
                s -= 2
        if d and r:
            dd = decode(fixed_r)
            if (d.value.year % 100, d.value.month, d.value.day) == (dd["yy"], dd["month"], dd["day"]):
                ag_dob, s = True, s + 4
            else:
                s -= 2
        if s > best_s:
            best, best_s = (n, r, d, fixed_r, ag_name, ag_dob), s

    n, r, d, fixed_r, ag_name, ag_dob = best
    out = Reading(agree_name=ag_name, agree_dob=ag_dob)
    if n:
        first, middle, last = n.value
        if ag_name:
            # the RAMQ only proves the first 3 letters + initial; the rest rests on the OCR
            nc = 0.75 + 0.2 * n.conf
        else:
            nc = min(0.8, 0.3 + 0.1 * n.score) * (0.7 + 0.3 * n.conf)
            if order_is_ambiguous(n.text) and not n.text.startswith(("NOM", "Nom")):
                nc = min(nc, 0.45)  # can't tell family name from given names: don't trust alone
        out.first = Field(nice_case(first), nc)
        out.middle = Field(nice_case(middle), nc * 0.9 if middle else nc)
        out.last = Field(nice_case(last), nc)
    if d and (ag_dob or d.score >= 1.5):  # an unlabelled, unconfirmed date is probably not the DOB
        out.dob = Field(d.value, 0.98 if ag_dob else min(0.6, 0.2 + 0.1 * d.score))
    if r:
        rc = 0.3 + 0.3 * ag_name + 0.38 * ag_dob  # a RAMQ nothing confirms is weak evidence
        out.ramq = Field(fixed_r, rc, ["letters match name"] * ag_name + ["digits match DOB"] * ag_dob)
        out.sex = decode(fixed_r)["sex"]
    return out
