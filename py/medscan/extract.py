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
import math
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
            out.append((b, t, c, [c] * len(t)))
            continue
        sp = ctc.spans(frames)
        if "".join(s[0] for s in sp) != t:
            out.append((b, t, c, [c] * len(t)))
            continue
        # how sure the reader was of each letter: its best probability over the steps it spans
        cp = [max(p for ch, p in sum((frames[k] for k in range(t0, t1 + 1)), []) if ch == s)
              for s, t0, t1 in sp]
        edits = []  # (score, start, end, new_text)
        for m in RAMQ_SPAN.finditer(t):
            g, i, j = m[1], m.start(1), m.end(1)
            if sum(ch.isalpha() for ch in g[:5]) < 2 or sum(ch.isdigit() for ch in g) < 4:
                continue
            t0, t1 = ctc.window(frames, sp, i, j - 1)
            txt, total, n, probs = ctc.decode(frames, ctc.RAMQ, t0, t1, with_probs=True)
            if txt:
                r = re.sub(r"[\s\-]", "", txt).upper()
                if is_valid(r) and total / max(1, n) > -0.8:
                    edits.append((total / n, i, j, txt.upper(), probs))
                    paths[r] = (frames, t0, t1, total)
        for m in DATE_SPAN.finditer(t):
            g, i, j = m[1], m.start(1), m.end(1)
            if sum(ch.isdigit() for ch in g) < 4:
                continue
            t0, t1 = ctc.window(frames, sp, i, j - 1)
            best = None
            for pat in (ctc.DATE_YMD, ctc.DATE_DMY):
                txt, total, n, probs = ctc.decode(frames, pat, t0, t1, with_probs=True)
                if txt and parse_dates(txt) and (best is None or total > best[0]):
                    best = (total, n, txt, probs)
            if best and best[0] / max(1, best[1]) > -0.8:
                edits.append((best[0] / best[1], i, j, best[2], best[3]))
        # apply the best non-overlapping edits, right to left
        chosen = []
        for e in sorted(edits, key=lambda e: e[0], reverse=True):
            if all(e[2] <= o[1] or e[1] >= o[2] for o in chosen):
                chosen.append(e)
        for score, i, j, new, probs in sorted(chosen, key=lambda e: -e[1]):
            t = t[:i] + new + t[j:]
            cp = cp[:i] + (probs if len(probs) == len(new) else [math.exp(score)] * len(new)) + cp[j:]
        out.append((b, t, c, cp))
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
    items = sorted((_with_letter_probs(o) for o in items), key=lambda o: (o[0][1] + o[0][3]) / 2)
    used = [False] * len(items)
    out = []
    for i, (b, t, c, cp) in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        group = [(b, t, c, cp)]
        h = b[3] - b[1]
        cy = (b[1] + b[3]) / 2
        changed = True
        while changed:
            changed = False
            for j, (b2, t2, c2, cp2) in enumerate(items):
                if used[j] or abs((b2[1] + b2[3]) / 2 - cy) > 0.5 * h:
                    continue  # another line
                h2 = b2[3] - b2[1]
                x0, x1 = min(g[0][0] for g in group), max(g[0][2] for g in group)
                # a form label ('Nom:', 'DDN:') takes the value to its right even across a wide
                # gap and in another text size (values are often bold or handwritten)
                label = max(group, key=lambda g: g[0][2])[1].rstrip().endswith(":") and b2[0] >= x1 - 2
                if max(h, h2) > (2.5 if label else 1.4) * min(h, h2):
                    continue  # different text size: e.g. a header next to a sticker
                gap = max(b2[0] - x1, x0 - b2[2])
                if gap < (6 * h if label else 0.8 * h):  # ~ a word space; more = another column
                    group.append((b2, t2, c2, cp2))
                    used[j] = changed = True
        group.sort(key=lambda g: g[0][0])
        box = [min(g[0][0] for g in group), min(g[0][1] for g in group),
               max(g[0][2] for g in group), max(g[0][3] for g in group)]
        conf = sum(g[2] for g in group) / len(group)
        cps = []
        for g in group:
            cps += ([1.0] if cps else []) + list(g[3])  # the joining space is certain
        out.append((box, " ".join(g[1] for g in group), conf, cps))
    return out


def _with_letter_probs(o):
    """(bbox, text, conf[, letter_probs]) -> always 4 parts; without per-letter probabilities
    (other engines, tests) every letter gets the line's confidence."""
    b, t, c = o[:3]
    cp = o[3] if len(o) > 3 else None
    ok = isinstance(cp, list) and len(cp) == len(t) and all(isinstance(p, (int, float)) for p in cp)
    return (b, t, c, cp if ok else [c] * len(t))


def word_conf(r, word):
    """The reader's doubt about a word: the LOWEST probability among its letters in row r
    (one misread letter, 'Pierre' -> 'Peme', shows as one unsure letter; a line average
    hides it). Falls back to the row's confidence, lowered if the word isn't found as read
    (e.g. a digit was turned into a letter)."""
    t, c = r[1], r[2]
    cp = r[3] if len(r) > 3 else None
    if not cp or len(cp) != len(t):
        return c
    i = t.find(word)
    if i < 0:
        i = t.upper().find(word.upper())
    if i < 0:
        return c * 0.7
    return min(cp[i:i + len(word)]) if word else c


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
    """Cand.conf = the reader's confidence in the RAMQ's least certain DIGIT: the letters are
    checked against the name, but nothing checks the last two (administrative) digits, and
    nothing checks the date digits when no DOB is printed."""
    out = []
    for row in rs:
        b, t, c = row[:3]
        for r in find_ramq_loose(t):
            s = 1.0 + c
            if re.search(r"\b(NAM|RAMQ|RAMO|MALADIE|ASS)\b", fold(t)):
                s += 1.0
            out.append(Cand(r, s, t, digit_conf(row, r), b))
    return out


def digit_conf(row, ramq):
    """Lowest per-character probability among the 8 digits of `ramq` as written in the row
    ('TREM 8555 1512', 'TREM85551512', 'TREM-8555-1512'); the row confidence if not found."""
    t, c = row[1], row[2]
    cp = row[3] if len(row) > 3 else None
    m = re.search(rf"{ramq[:4]}[\s\-]?{ramq[4:8]}[\s\-]?{ramq[8:]}", t, re.I)
    if not m or not cp or len(cp) != len(t):
        return c * 0.8
    return min(p for ch, p in zip(t[m.start():m.end()], cp[m.start():m.end()]) if ch.isdigit())


def lines_count(boxes):
    """How many separate lines on the page: boxes whose vertical centres are within half a
    line height of each other are the same line (a labelled row and its bare value piece)."""
    centres = []
    for b in boxes:
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if not any(abs(cy - y) < 0.5 * max(h, hh) and (min(b[2], x1) - max(b[0], x0)) > -h
                   for y, hh, x0, x1 in centres):
            centres.append((cy, h, b[0], b[2]))
    return len(centres)


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


def _short_year_dob(yy, m, d):
    """A date written with a 2-digit year: the most recent century that isn't in the future."""
    return _valid_dob(2000 + yy, m, d) or _valid_dob(1900 + yy, m, d)


# between the parts of a date: '-', '/' or '.' (spaces around them allowed), or just a space
_SEP = r"(?:\s?[-/.]\s?|\s)"
# a RAMQ number's own digits are not a date, however the reader spaced them
_RAMQ_SPAN = re.compile(r"(?<![A-Z])[A-Z]{4}\s?-?\s?\d{2}\s?\d{2}\s?-?\s?\d{2}\s?\d{2}(?!\d)")


def parse_dates(text):
    """All plausible birth dates in a line, as (date, how_sure) pairs."""
    t = _digits_fixed(fold(text))
    found = []

    def spaced(m):  # "2002 12 05": only spaces between the parts is weaker evidence of a date
        return 0.0 if re.search(r"[-/.]", m[0]) else 0.15

    for m in re.finditer(rf"(?<!\d)(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)", t):
        x = _valid_dob(int(m[1]), int(m[2]), int(m[3]))
        if x:
            found.append((x, 1.0 - spaced(m)))
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}})(?!\d)", t):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        x = _valid_dob(y, b, a)  # Quebec: JJ/MM/AAAA
        if x:
            found.append((x, 0.9 - spaced(m)))
        x2 = _valid_dob(y, a, b)  # MM/DD/YYYY (English forms) is possible but less likely
        if x2 and x2 != x:
            found.append((x2, 0.4))
    # Short dates with a 2-digit year ("85-05-15", "85 05 15"): the order can't be told from the
    # date alone. Quebec health cards and forms write AA-MM-JJ; JJ-MM-AA and MM-JJ-AA also
    # exist. Every valid reading is offered with modest confidence, so on its own a short
    # date is not trusted: the RAMQ's digits settle which reading it is (see extract()).
    short = _RAMQ_SPAN.sub(" ", t)
    for m in re.finditer(rf"(?<!\d)(?<!\d[-/. ])(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2}})(?!\d)"
                         r"(?![-/. ]\d{1,2}(?!\d))", short):
        a, b, c = int(m[1]), int(m[2]), int(m[3])
        readings = [(c, b, a, 0.5), (c, a, b, 0.25)]          # JJ-MM-AA, MM-JJ-AA
        if len(m[1]) == 2:
            readings.insert(0, (a, b, c, 0.6))                # AA-MM-JJ
        for yy, mo, dd, sure in readings:
            x = _short_year_dob(yy, mo, dd)
            if x and all(x != f[0] for f in found):
                found.append((x, sure))
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
        b, t, c = r[:3]
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


NAME_LABELS = ["NOM", "PATIENT", "PATIENTE", "NOM DU PATIENT", "NAME", "USAGER", "BENEFICIAIRE", "NOM COMPLET"]
# labels in front of the patient's name on forms, with or without ':' ("Nom complet TANGUAY, Aurélie")
NAME_LABEL_RE = re.compile(
    r"^\s*(nom\s+complet|nom\s+du\s+patient|nom\s+de\s+l'usager|nom\s+et\s+pr[eéèêë]noms?|nom\s*,\s*pr[eéèêë]noms?"
    r"|patiente?|usag[eè]re?|b[ée]n[ée]ficiaire|nom)(?![^\W\d_])\s*[:.\-]?\s*", re.I)
GIVEN_LABELS = ["PRENOM", "PRENOMS"]
HONORIFICS = {"MME", "M", "M.", "MR", "MRS", "MS", "MLLE", "MADAME", "MONSIEUR"}
STOP = set("""HOPITAL CENTRE CHU CHUM CUSM CISSS CIUSSS INSTITUT CLINIQUE GMF PHARMACIE CARTE NOTE NOTES
EVOLUTION ORDONNANCE ORDONNANCES FEUILLE CONSULTATION URGENCE DATE HEURE ALLERGIE ALLERGIES PLAN RX DOSSIER
SEXE NAM RAMQ DDN PERMIS DR DRE DOCTEUR MEDECIN QTE REN RUE BOUL AV TEL MG MCG ML COMPRIME COMPRIMES
CAPSULE SIGNATURE TELEPHONE ADRESSE EXP SERVICE UNITE CHAMBRE LIT ETAGE LABORATOIRE RADIOLOGIE
QUEBEC MONTREAL LAVAL""".split())
TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "4": "A", "3": "E", "7": "T", "2": "Z"}


def _clean_name_tokens(text, min_words=2):
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
    if not min_words <= len(words) <= 5:
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
        b, t, c = r[:3]
        ft = fold(t)
        label_bonus = 0.0
        body = t
        lab = NAME_LABEL_RE.match(t)
        if lab:
            body = t[lab.end():].strip()
            if not body:
                continue
            if fold(lab[1]) == "NOM" and "," not in body and len(body.split()) == 1:
                last_lab = (body, c, b, r)  # 'Nom: SAVOIE' -> given names are on the 'Prénom' line
                continue
            pieces = body.split()
            if fold(lab[1]) == "NOM" and "," not in body and body.isupper() and min(map(len, pieces)) <= 4:
                # 'Nom: LAROU CHE': block capitals with wide letter spacing came back in pieces;
                # also try them joined as one family name, to pair with the 'Prénom' line
                last_lab = ("".join(pieces), c, b, r)
            label_bonus = 1.5
            m = None
        else:
            m = re.match(r"\s*([^\W\d_][^\W\d_ .]*(?: [^\W\d_]+)?)\s*([:.])\s*(.*)$", t)
        if m:
            if fuzzy_in(m[1], NAME_LABELS):
                label_bonus, body = 1.5, m[3]
            elif fuzzy_in(m[1], GIVEN_LABELS):
                if m[3].strip():
                    given_lab = (m[3], c, b, r)
                continue
            elif m[2] == ":" and len(m[1]) <= 12 and not fuzzy_in(m[1], list(HONORIFICS)):
                continue  # 'Allergie : ...', 'Plan : ...' -> some other field ('DUBÉ. Marie' is not)
        first_word = ft.split()[0] if ft.split() else ""
        if fuzzy_in(first_word, GIVEN_LABELS, 0.8):
            if len(t.split()) > 1:  # a lone 'PRÉNOM' piece must not erase the full line
                given_lab = (t.split(None, 1)[1], c, b, r)
            continue
        up = above(rs, r)
        if up and fuzzy_in(fold(up[1]).strip(" :"), NAME_LABELS):
            label_bonus = 1.0
        if re.search(r"\b(DRE?|PERMIS|MEDECIN)\b", ft):
            continue  # the physician, not the patient
        toks = _clean_name_tokens(body)
        if not toks:
            continue
        wc = min(word_conf(r, w) for w in toks if w != ",")  # its least certain letter
        for (first, middle, last), prior in _splits(toks):
            out.append(Cand((first, middle, last), prior + label_bonus + wc * 0.5, t, wc, b))
    if last_lab and given_lab:  # card / form layout: 'NOM LEMIEUX' / 'PRÉNOM GENEVIÈVE LUCIE'
        g = _clean_name_tokens(given_lab[0], min_words=1) or []
        g = [w for w in g if w != ","]
        lt = _clean_name_tokens(last_lab[0], min_words=1)
        if g and lt:
            wc = min([word_conf(last_lab[3], w) for w in lt] + [word_conf(given_lab[3], w) for w in g])
            out.append(Cand((g[0], " ".join(g[1:]), last_lab[0].strip()), 2.5 + wc * 0.5,
                            f"{last_lab[0]} / {given_lab[0]}", wc, last_lab[2]))
    out += stacked_names(rs)
    return out


def stacked_names(rs):
    """Cards often print the family name and the given names on two unlabelled lines:
        TREMBLAY
        MARIE-ÈVE ANNE
    Pairs of left-aligned, close, short all-capital name lines become candidates in both
    orders (family name first is more common); the RAMQ cross-check settles which is which."""
    lines = []
    for r in rs:
        b, t, c = r[:3]
        if t.rstrip().endswith(":"):
            continue  # a label ('DDN:' misread as 'DON:'), not a name
        toks = _clean_name_tokens(t, min_words=1)
        if toks and "," not in toks and len(toks) <= 3 and all(w.isupper() and len(w) >= 2 for w in toks):
            lines.append((b, toks, min(word_conf(r, w) for w in toks), t))
    lines.sort(key=lambda l: l[0][1])
    out = []
    # every pair, not just neighbours in top-to-bottom order: other text at the same height
    # elsewhere on the page ('MSSS' logo on the left) must not hide the pair
    pairs = [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]]
    for (b1, t1, c1, s1), (b2, t2, c2, s2) in pairs:
        h = max(b1[3] - b1[1], b2[3] - b2[1])
        if abs(b1[0] - b2[0]) > 0.8 * h or not (-0.4 * h <= b2[1] - b1[3] < 1.2 * h):
            continue
        conf = min(c1, c2)  # least certain letter of either line
        if len(t1) == 1:  # family name above (hyphenated names are one token)
            out.append(Cand((t2[0], " ".join(t2[1:]), t1[0]), 1.2 + conf * 0.5, f"{s1} / {s2}", conf, b1))
        if len(t2) == 1:  # given name above
            out.append(Cand((t1[0], " ".join(t1[1:]), t2[0]), 0.6 + conf * 0.5, f"{s1} / {s2}", conf, b1))
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
    dob_source: str = "read"  # "read" off the page, or worked out "ramq" (no DOB printed / readable)
    name_letter_conf: float = 0.0  # reader's confidence in the chosen name's least certain letter
    name_lines: int = 0            # separate lines on the page that read exactly this name
    ramq_digit_conf: float = 0.0   # reader's confidence in the RAMQ's least certain digit
    ramq_lines: int = 0            # separate lines on the page that read exactly this RAMQ

    def as_dict(self):
        return {"first": self.first.value, "middle": self.middle.value, "last": self.last.value,
                "dob": self.dob.value.isoformat() if self.dob.value else None, "ramq": self.ramq.value,
                "sex": self.sex, "agree_name": self.agree_name, "agree_dob": self.agree_dob,
                "dob_source": self.dob_source, "name_letter_conf": round(self.name_letter_conf, 3),
                "name_lines": self.name_lines, "ramq_digit_conf": round(self.ramq_digit_conf, 3),
                "ramq_lines": self.ramq_lines,
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
    all_names = name_candidates(rs)
    names = _top(all_names, 8) + [None]
    all_ramqs = ramq_candidates(rs)
    ramqs = _top(all_ramqs, 4) + [None]
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
        out.name_letter_conf, out.name_lines = n.conf, 1
        if ag_name:  # several readings of the same name on the page: keep the cleanest
            first, middle, last, out.name_letter_conf, out.name_lines = name_consensus(
                all_names, fixed_r[:4], (first, middle, last))
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
        same = [c for c in all_ramqs if c.value in (r.value, fixed_r)]
        out.ramq_digit_conf = max(c.conf for c in same)
        out.ramq_lines = lines_count([c.bbox for c in same])
        if out.dob.value is None and ag_name:
            dob_from_ramq(out, fixed_r)
    return out


def confident_still(rd, t_name=0.85, t_digits=0.9):
    """For ONE still image (no further views to vote with): the parts that no cross-check
    covers must be backed by the page itself, either read identically on 2+ separate lines,
    or read with high confidence in their least certain character:
      - the name beyond the RAMQ's letters (e.g. 'Patncia' vs 'Patricia'),
      - the RAMQ's 2 administrative digits, and all 8 digits when the DOB came from the RAMQ.
    The thresholds are chosen on a training split (see eval_external.py --sweep)."""
    name_ok = rd.name_lines >= 2 or rd.name_letter_conf >= t_name
    ramq_ok = rd.ramq_lines >= 2 or rd.ramq_digit_conf >= t_digits
    return name_ok and ramq_ok


def still_ok(rd, t_name=0.85, t_digits=0.9):
    """Can ONE still image be proposed on its own (no further views to vote with)?
    Returns (True, "") or (False, why not). Every field must be there, the name and the
    birth date must agree with the RAMQ, and confident_still() must hold."""
    if not rd.ramq.value:
        return False, "no RAMQ read"
    if not rd.dob.value:
        return False, "no DOB read"
    if not (rd.last.value and rd.first.value):
        return False, "no name read"
    if not rd.agree_name:
        return False, "name doesn't match RAMQ letters"
    if not (rd.agree_dob or rd.dob_source == "ramq"):
        return False, "DOB doesn't match RAMQ digits"
    if not confident_still(rd, t_name, t_digits):
        name_sure = rd.name_lines >= 2 or rd.name_letter_conf >= t_name
        return False, "not sure enough: " + ("RAMQ digits" if name_sure else "name")
    return True, ""


def _squash(s):
    return re.sub(r"\s+", "", fold(s))


def _close(a, b):
    """Same name read twice? Equal once spaces are removed ('SÉBAS TIEN' = 'Sébastien'),
    or within ~1 edit per 6 letters ('Genevneve' ~ 'GENEVIEVE')."""
    a, b = _squash(a), _squash(b)
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85 and abs(len(a) - len(b)) <= 1


def name_consensus(cands, code, chosen):
    """A page often shows the name more than once (ID block, 'Nom complet:', 'Usager'). Among
    the readings that match the RAMQ letters, group the ones that are the same name and keep
    the cleanest: fewest stray spaces (letter-spaced block capitals split into pieces),
    then the highest OCR confidence, then the one with accents (an accent is rarely invented)."""
    pool = []
    for c in cands:
        f, m, l = c.value
        if name_code(l, f) == code:
            pool.append((f, m, l, c.conf, c.bbox))
    if len(pool) < 2:
        return chosen + _evidence(pool, chosen)

    def pick(values):
        """values: (string, conf, bbox) -> the cleanest string of the best-supported group."""
        groups = []
        for v in values:
            for g in groups:
                if _close(g[0][0], v[0]):
                    g.append(v)
                    break
            else:
                groups.append([v])
        # support: each physical line counts once (a labelled row and its bare piece are one line)
        best = max(groups, key=lambda g: (lines_count([b for _, _, b in g]), max(c for _, c, _ in g)))
        min_sp = min(s.count(" ") for s, _, _ in best)

        def accent_wins(s):  # 'Marie-Ève' over 'MARIE-EVE' (same letters, accents kept)
            return s.upper() != fold(s) and any(_squash(o) == _squash(s) and o != s for o, _, _ in best)

        return max(best, key=lambda v: (-(v[0].count(" ") - min_sp), accent_wins(v[0]), v[1]))[0]

    last = pick([(l, c, b) for f, m, l, c, b in pool])
    given = pick([(" ".join(x for x in (f, m) if x), c, b) for f, m, l, c, b in pool])
    gw = given.split()
    name = (gw[0], " ".join(gw[1:]), last) if gw else chosen
    if name_code(name[2], name[0]) != code:  # the two picks must still make a name that fits the RAMQ
        name = chosen
    return name + _evidence(pool, name)


def _evidence(pool, name):
    """How well-supported is this exact spelling? (least-certain-letter confidence of the best
    reading of it, number of separate lines that read it; letters compared without spaces)."""
    key = (_squash(name[2]), _squash(name[0] + name[1]))
    same = [(c, b) for f, m, l, c, b in pool if (_squash(l), _squash(f + m)) == key]
    return (max(c for c, _ in same) if same else 0.0), lines_count([b for _, b in same])


def dob_from_ramq(out, ramq):
    """No DOB printed / readable, but a RAMQ whose letters match the name: its digits ARE the
    birth date (YY, MM +50 for women, DD). Only the century is not written: take the most
    recent one that isn't in the future (a 2-digit year <= this year could also be a
    centenarian; that case is flagged)."""
    d = decode(ramq)
    for century in (2000, 1900):
        try:
            x = date(century + d["yy"], d["month"], d["day"])
        except ValueError:
            return
        if x <= TODAY:
            checks = ["from RAMQ"]
            if century == 2000 and TODAY.year - (1900 + d["yy"]) <= 110:
                checks.append("century assumed")
            out.dob = Field(x, 0.5, checks)
            out.dob_source = "ramq"
            return
