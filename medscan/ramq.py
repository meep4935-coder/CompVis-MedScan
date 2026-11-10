"""
Quebec health insurance number (RAMQ "NAM", numéro d'assurance maladie).

Format: 4 letters + 8 digits, usually printed "TREM 8555 1512".
    TRE   first 3 letters of the family name (accents, spaces, hyphens, apostrophes removed;
          padded with X when shorter than 3 letters)            <- assumption, verify
    M     first letter of the first given name
    85    year of birth (2 digits)
    55    month of birth, +50 for women  (05 -> 55)
    15    day of birth
    12    2 administrative digits (not derivable from the person)

Because the letters and digits encode the name, sex and date of birth, the three
fields we want to read cross-check each other: if OCR reads a RAMQ number that
disagrees with the name or DOB it read, something was misread.
"""

import re
import unicodedata
from datetime import date

RAMQ_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{4})[\s\-]?(\d{4})[\s\-]?(\d{4})(?![A-Z0-9])", re.I)
# OCR often confuses these; a RAMQ number has letters ONLY in positions 1-4, digits ONLY in 5-12
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7"}


def letters_only(s):
    """'Côté-D'Amour' -> 'COTEDAMOUR' (strip accents, keep A-Z)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s.upper() if "A" <= c <= "Z")


def name_code(last, first):
    """The 4 letters of a RAMQ number."""
    return (letters_only(last) + "XXX")[:3] + letters_only(first)[:1]


def make_ramq(last, first, dob: date, sex: str, admin: int) -> str:
    month = dob.month + (50 if sex.upper() == "F" else 0)
    return f"{name_code(last, first)}{dob.year % 100:02d}{month:02d}{dob.day:02d}{admin % 100:02d}"


def format_ramq(r, style="spaced"):
    """'TREM85551512' -> 'TREM 8555 1512' | 'TREM85551512' | 'TREM-8555-1512'."""
    if style == "compact":
        return r
    sep = "-" if style == "dash" else " "
    return f"{r[:4]}{sep}{r[4:8]}{sep}{r[8:]}"


def fix_confusions(candidate):
    """Apply the letters-then-digits rule to a 12-char OCR read: '7REM8S551S12' -> 'TREM85551512'."""
    c = re.sub(r"[\s\-]", "", candidate.upper())
    if len(c) != 12:
        return None
    head = "".join(TO_LETTER.get(ch, ch) for ch in c[:4])
    tail = "".join(TO_DIGIT.get(ch, ch) for ch in c[4:])
    out = head + tail
    return out if re.fullmatch(r"[A-Z]{4}\d{8}", out) else None


def find_ramq(text):
    """All RAMQ-looking numbers in a block of text, normalised to 12 chars."""
    return ["".join(m.groups()).upper() for m in RAMQ_RE.finditer(text)]


def decode(r):
    """Split a RAMQ number into what it says about the person."""
    yy, mm, dd = int(r[4:6]), int(r[6:8]), int(r[8:10])
    sex = "F" if mm > 50 else "M"
    month = mm - 50 if mm > 50 else mm
    return {"letters": r[:4], "yy": yy, "month": month, "day": dd, "sex": sex, "admin": r[10:12]}


def is_valid(r):
    """Format + plausible date (some century must make it a real date)."""
    if not re.fullmatch(r"[A-Z]{4}\d{8}", r or ""):
        return False
    d = decode(r)
    for century in (1900, 2000):
        try:
            date(century + d["yy"], d["month"], d["day"])
            return True
        except ValueError:
            pass
    return False


def check(r, last=None, first=None, dob: date = None, sex=None):
    """Compare a RAMQ number against other fields we read. Returns a list of problems
    (empty list = everything we know agrees)."""
    if not is_valid(r):
        return ["invalid RAMQ format/date"]
    d = decode(r)
    problems = []
    if last is not None and first is not None and name_code(last, first) != d["letters"]:
        problems.append(f"letters {d['letters']} != name {name_code(last, first)}")
    if dob is not None and (dob.year % 100, dob.month, dob.day) != (d["yy"], d["month"], d["day"]):
        problems.append(f"date in RAMQ != DOB {dob.isoformat()}")
    if sex is not None and sex.upper() != d["sex"]:
        problems.append(f"sex in RAMQ ({d['sex']}) != {sex}")
    return problems
