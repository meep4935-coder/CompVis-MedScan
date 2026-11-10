"""
Score medscan.extract on cached OCR results (run eval_ocr.py first).

    python eval_extract.py --docs data/docs_eval            # clean pages
    python eval_extract.py --docs data/docs_eval --camera   # webcam-like pages
    python eval_extract.py --show-errors 15

For each field on each page: right, wrong (a confident mistake: the dangerous kind)
or missing (left empty: safe, the live program just keeps reading).
A field the page doesn't show (e.g. no RAMQ on a pharmacy label) counts as
'invented' if the extractor fills it anyway.
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from medscan.extract import extract


def norm(s):
    return (s or "").casefold().strip()


def score_page(truth, items, stats, errors, key):
    p = truth["patient"]
    on_page = {f["field"] for f in truth["fields"]}
    has_name = bool(on_page & {"name", "last_name", "given_names"})
    rd = extract(items)
    got = rd.as_dict()
    expect = {
        "last": (has_name, norm(p["last"]), norm(got["last"])),
        "first": (has_name, norm(p["first"]), norm(got["first"])),
        "middle": (has_name and _middle_shown(truth), norm(p["middle"]), norm(got["middle"])),
        "dob": ("dob" in on_page, p["dob"], got["dob"]),
        "ramq": ("ramq" in on_page, p["ramq"], got["ramq"]),
    }
    page_ok = True
    for f, (shown, want, have) in expect.items():
        if not shown:
            if have and f != "middle":
                stats[f"{f}_invented"] += 1
                errors.append((key, f, "not on page", have))
            continue
        stats[f"{f}_n"] += 1
        if not have and want:
            stats[f"{f}_missing"] += 1
            page_ok = False
        elif have == want:
            stats[f"{f}_right"] += 1
        else:
            stats[f"{f}_wrong"] += 1
            page_ok = False
            errors.append((key, f, want, have))
    if {"dob", "ramq"} <= on_page and has_name:
        stats["full_n"] += 1
        stats["full_ok"] += page_ok
        stats["full_agree"] += got["agree_name"] and got["agree_dob"]
        stats["full_agree_ok"] += got["agree_name"] and got["agree_dob"] and page_ok


def _middle_shown(truth):
    """Is the middle name actually written on the page? (some layouts drop it)"""
    m = truth["patient"]["middle"]
    if not m:
        return True  # then the right answer is 'no middle name'
    return any(m.casefold() in f["text"].casefold() for f in truth["fields"] if f["field"] in ("name", "given_names"))


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   -  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs_eval")
    ap.add_argument("--engine", default="rapid")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--show-errors", type=int, default=10)
    args = ap.parse_args()
    docs = Path(args.docs)
    cache = json.loads((docs / f"ocr_{args.engine}{'_camera' if args.camera else ''}.json").read_text(encoding="utf-8"))
    stats, errors = defaultdict(int), []
    by_type = defaultdict(lambda: defaultdict(int))
    for k in sorted(x for x in cache if not x.startswith("_")):
        truth = json.loads((docs / f"{k}.json").read_text(encoding="utf-8"))
        score_page(truth, cache[k], stats, errors, k)
        score_page(truth, cache[k], by_type[truth["type"]], [], k)

    print(f"{args.engine} {'camera-like' if args.camera else 'clean'} pages")
    print(f"{'field':8} {'right':>7} {'wrong':>7} {'missing':>8} {'invented':>9}   (n)")
    for f in ("last", "first", "middle", "dob", "ramq"):
        n = stats[f"{f}_n"]
        print(f"{f:8} {pct(stats[f + '_right'], n):>7} {pct(stats[f + '_wrong'], n):>7} "
              f"{pct(stats[f + '_missing'], n):>8} {stats[f + '_invented']:>9}   ({n})")
    print(f"pages with all 3 fields: fully right {pct(stats['full_ok'], stats['full_n'])} of {stats['full_n']}; "
          f"all-agree {pct(stats['full_agree'], stats['full_n'])}, "
          f"right when all agree {pct(stats['full_agree_ok'], stats['full_agree'])}")
    print("by type (full pages right):", {t: pct(s['full_ok'], s['full_n']).strip() for t, s in sorted(by_type.items())})
    for e in errors[:args.show_errors]:
        print("  ", e)


if __name__ == "__main__":
    main()
