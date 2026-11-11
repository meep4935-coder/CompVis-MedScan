"""
Simulate the live loop: read -> vote -> propose -> one-tap accept/reject -> keep reading.

For every document, up to --views different webcam-like views are made (each with its own
resolution, blur, noise, lighting and JPEG quality, like a page moving in front of a camera).
They are OCR'd, run through medscan.extract, and fed one by one to a medscan.vote.Voter.
A simulated physician accepts a correct proposal and rejects a wrong one.

    python eval_live.py --docs data/docs_eval --views 8

Reported:
    accepted        the right patient record was proposed and accepted
    wrong proposals proposals the physician had to reject (the number to keep near zero)
    no proposal     never confident enough within the views given (safe: it keeps reading)
    views needed    distinct views until the accepted proposal
Pages that don't show all three fields (pharmacy labels, some notes) must never be proposed
on their own: they need a second document of the same patient.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import cv2

from eval_extract import _middle_shown
from eval_ocr import ENGINES, camera_like
from medscan.extract import extract
from medscan.vote import Voter


def truth_record(p):
    return {"last": p["last"].casefold(), "first": p["first"].casefold(), "middle": p["middle"].casefold(),
            "dob": p["dob"], "ramq": p["ramq"]}


def same(rec, want):
    got = {"last": (rec["last"] or "").casefold(), "first": (rec["first"] or "").casefold(),
           "middle": (rec["middle"] or "").casefold(),
           "dob": rec["dob"].isoformat() if isinstance(rec["dob"], date) else rec["dob"], "ramq": rec["ramq"]}
    return got == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs_eval")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--engine", default="rapidp", help="rapid (plain) or rapidp (with probabilities)")
    args = ap.parse_args()

    docs = Path(args.docs)
    keys = sorted(p.stem for p in docs.glob("doc_*.json"))
    if args.limit:
        keys = keys[:args.limit]
    cache_file = docs / f"ocr_{args.engine}_views.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    run = None
    for i, k in enumerate(keys, 1):
        for v in range(args.views):
            ck = f"{k}/{v}"
            if ck in cache:
                continue
            run = run or ENGINES[args.engine]()
            img, s = camera_like(cv2.imread(str(docs / f"{k}.jpg")), seed=int(k.split("_")[1]) * 100 + v)
            cache[ck] = [([x / s for x in it[0]], *it[1:]) for it in run(img)]
        print(f"\rOCR {i}/{len(keys)}", end="", flush=True)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print()

    full = {"n": 0, "accepted": 0, "wrong": 0, "none": 0, "views": [], "after_reject": 0, "hint": 0}
    partial = {"n": 0, "proposed": 0}
    wrong_examples = []
    for k in keys:
        truth = json.loads((docs / f"{k}.json").read_text(encoding="utf-8"))
        on_page = {f["field"] for f in truth["fields"]}
        complete = {"ramq", "dob"} <= on_page and bool(on_page & {"name", "last_name", "given_names"})
        want = truth_record(truth["patient"])
        if not _middle_shown(truth):
            want["middle"] = ""  # not written on this page: 'no middle name' is the right read
        voter = Voter(min_views=args.min_views)
        used, done = 0, False
        for v in range(args.views):
            used += 1
            rec = voter.add(extract(cache[f"{k}/{v}"]))
            if not rec:
                continue
            if not complete:
                partial["proposed"] += 1
                wrong_examples.append((k, "proposed from an incomplete page", rec))
                done = True
                break
            if same(rec, want):
                voter.accept()
                full["accepted"] += 1
                full["views"].append(used)
                full["after_reject"] += bool(voter.rejected)
                done = True
                break
            full["wrong"] += 1
            wrong_examples.append((k, want, {f: str(x) for f, x in rec.items()}))
            if voter.reject() == "need_better_view":
                full["hint"] += 1
                done = True
                break
        if complete:
            full["n"] += 1
            if not done:
                full["none"] += 1
        else:
            partial["n"] += 1

    n = full["n"]
    vs = sorted(full["views"])
    print(f"Pages with name + RAMQ + DOB: {n}, up to {args.views} views each, min {args.min_views} views per field")
    print(f"  accepted (right)      {full['accepted']:4d}  {100 * full['accepted'] / n:5.1f}%"
          f"   ({full['after_reject']} of them after a reject)")
    print(f"  wrong proposals       {full['wrong']:4d}   (each one = one 'reject' tap)")
    print(f"  asked for better view {full['hint']:4d}")
    print(f"  no proposal yet       {full['none']:4d}  {100 * full['none'] / n:5.1f}%")
    if vs:
        print(f"  views needed: median {vs[len(vs) // 2]}, max {vs[-1]}")
    print(f"Pages missing a field: {partial['n']}, proposed on their own: {partial['proposed']} (should be 0)")
    for e in wrong_examples[:10]:
        print("   ", e)


if __name__ == "__main__":
    main()
