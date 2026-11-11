"""
Score the reading pipeline on an external RAMQ dataset: a folder with manifest.jsonl rows like
    {"id", "split", "image_path", "ramq", "patient_name": "LAST, First", "decoy_nams": [...],
     "difficulty", "header_mode", "template"}
(e.g. github.com/OuifakElWarrari/ramq-synthetic-dataset, cloned into data/external/).

    python eval_external.py --dataset data/external/ramq-synthetic-dataset/output --split test

A still image can't be re-scanned, so each one gets a single decision:
    PROPOSE  name + RAMQ + DOB all read and all agreeing (the bar the live loop needs before it
             shows a patient) -> right or WRONG (the dangerous kind)
    SKIP     anything less, with the reason. 'readable' says whether the true RAMQ appears
             anywhere in what the reader saw: if not, the image defeated the OCR (live, the camera
             would keep scanning); if yes, the extraction logic missed it (something to fix).
The true DOB isn't in the manifest; the RAMQ encodes it (YY MM(+50 for women) DD), so the DOB
read is checked against that.
One row per image goes to <dataset>/medscan_report_<split>.csv for later diagnosis.
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from medscan.extract import confident_still, extract, refine
from medscan.ramq import decode, find_ramq_loose


def truth_of(row):
    last, _, given = row["patient_name"].partition(",")
    d = decode(row["ramq"])
    return {"ramq": row["ramq"], "last": last.strip(), "first": given.strip().split()[0] if given.strip() else "",
            "yy": d["yy"], "month": d["month"], "day": d["day"], "sex": d["sex"]}


def judge(rd, t, decoys, gate=None):
    """gate = (t_name, t_digits) for medscan.extract.confident_still, or None (no gate)."""
    got = rd.as_dict()
    name_ok = (got["last"] or "").casefold() == t["last"].casefold() and \
              (got["first"] or "").casefold() == t["first"].casefold()
    from medscan.extract import fold
    name_accent_only = not name_ok and fold(got["last"] or "") == fold(t["last"]) and \
        fold(got["first"] or "") == fold(t["first"])
    dob = rd.dob.value
    dob_ok = bool(dob) and (dob.year % 100, dob.month, dob.day) == (t["yy"], t["month"], t["day"])
    ramq_ok = got["ramq"] == t["ramq"]
    complete = bool(got["last"] and got["first"] and got["ramq"] and dob)
    from_ramq = got["dob_source"] == "ramq"
    agree = complete and got["agree_name"] and (got["agree_dob"] or from_ramq)
    all_ok = name_ok and dob_ok and ramq_ok
    ungated = ("right" if all_ok else "WRONG") if agree else "skip"
    if agree and gate and not confident_still(rd, *gate):
        outcome = "skip"
        name_sure = rd.name_lines >= 2 or rd.name_letter_conf >= gate[0]
        reason = "not sure enough: " + ("RAMQ digits" if name_sure else "name")
    elif agree:
        outcome = ungated
        reason = "" if outcome == "right" else ",".join(
            f for f, ok in (("name", name_ok), ("ramq", ramq_ok), ("dob", dob_ok)) if not ok)
    else:
        outcome = "skip"
        if not got["ramq"]:
            reason = "no RAMQ read"
        elif not dob:
            reason = "no DOB read"
        elif not got["last"]:
            reason = "no name read"
        elif not got["agree_name"]:
            reason = "name doesn't match RAMQ letters"
        else:
            reason = "DOB doesn't match RAMQ digits"
    return {"outcome": outcome, "reason": reason, "ramq_got": got["ramq"] or "", "ramq_ok": ramq_ok,
            "decoy_chosen": got["ramq"] in decoys, "name_got": f"{got['last']}, {got['first']}",
            "name_ok": name_ok, "dob_got": dob.isoformat() if dob else "", "dob_ok": dob_ok,
            "dob_from_ramq": from_ramq, "name_accent_only": name_accent_only,
            "name_letter_conf": got["name_letter_conf"], "name_lines": got["name_lines"],
            "ramq_digit_conf": got["ramq_digit_conf"], "ramq_lines": got["ramq_lines"],
            "outcome_ungated": ungated}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/external/ramq-synthetic-dataset/output")
    ap.add_argument("--split", default="test", help="train / val / test / all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--t-name", type=float, default=0.85, help="still-image gate: name letter confidence")
    ap.add_argument("--t-digits", type=float, default=0.9, help="still-image gate: RAMQ digit confidence")
    ap.add_argument("--no-gate", action="store_true", help="propose whenever the fields agree")
    ap.add_argument("--no-detect", dest="detect", action="store_false",
                    help="skip the paper-detector check")
    args = ap.parse_args()

    ds = Path(args.dataset)
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]

    cache_file = ds.parent / f"medscan_ocr_{args.split}.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    reader = detector = None
    t0, n_new = time.time(), 0
    report = []
    for i, row in enumerate(rows, 1):
        img = None
        if row["id"] not in cache:
            if reader is None:
                from medscan.ocr import Reader
                reader = Reader()
            img = cv2.imread(str(ds / row["image_path"]))
            cache[row["id"]] = reader(img)
            n_new += 1
            if n_new % 10 == 0:
                cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        items = cache[row["id"]]
        t = truth_of(row)
        gate = None if args.no_gate else (args.t_name, args.t_digits)
        res = judge(extract(items), t, set(row.get("decoy_nams", [])), gate)
        fixed, _ = refine(items)
        res["readable"] = t["ramq"] in find_ramq_loose(" ".join(it[1] for it in fixed))
        if args.detect:
            if detector is None:
                from ultralytics import YOLO
                detector = YOLO("models/paper_seg.pt")
            img = img if img is not None else cv2.imread(str(ds / row["image_path"]))
            r = detector.predict(img, imgsz=416, conf=0.5, classes=[0], verbose=False)[0]
            area = 0.0
            if r.masks is not None and len(r.masks.xy):
                area = max(cv2.contourArea(p.astype("float32")) for p in r.masks.xy) / (img.shape[0] * img.shape[1])
            res["page_found"] = area > 0.3
        report.append({"id": row["id"], "difficulty": row["difficulty"], "header_mode": row["header_mode"],
                       "template": row["template"], "ramq_true": t["ramq"], **res})
        print(f"\r{i}/{len(rows)}", end="", flush=True)
    print()
    cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if n_new:
        print(f"OCR: {(time.time() - t0) / n_new:.1f} s/page")

    out = ds.parent / f"medscan_report_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    n = len(report)
    oc = Counter(r["outcome"] for r in report)
    print(f"\n{n} images ({args.split})")
    print(f"  proposed & right   {oc['right']:4d}  {100 * oc['right'] / n:5.1f}%")
    print(f"  proposed & WRONG   {oc['WRONG']:4d}  {100 * oc['WRONG'] / n:5.1f}%")
    wrong = [r for r in report if r["outcome"] == "WRONG"]
    print(f"    of the wrong ones, only an accent off in the name: {sum(r['name_accent_only'] for r in wrong)}")
    prop = [r for r in report if r["outcome"] != "skip"]
    print(f"    of the proposals, DOB worked out from the RAMQ (none readable): "
          f"{sum(r['dob_from_ramq'] for r in prop)}")
    print(f"  skipped            {oc['skip']:4d}  {100 * oc['skip'] / n:5.1f}%")
    sk = [r for r in report if r["outcome"] == "skip"]
    unsure = [r for r in sk if r["reason"].startswith("not sure")]
    print(f"    unreadable (true RAMQ not in OCR text):  {sum(not r['readable'] for r in sk)}")
    print(f"    deliberate: fields agree but not sure:   {len(unsure)}"
          f"  (would have been {sum(r['outcome_ungated'] == 'right' for r in unsure)} right,"
          f" {sum(r['outcome_ungated'] == 'WRONG' for r in unsure)} wrong)")
    print(f"    readable, a field not found (logic gap): "
          f"{sum(r['readable'] and not r['reason'].startswith('not sure') for r in sk)}")
    for reason, c in Counter(r["reason"] for r in sk).most_common():
        print(f"      {reason:34} {c}")
    print(f"  RAMQ alone right (their benchmark)  {sum(r['ramq_ok'] for r in report)}/{n};"
          f" decoy picked: {sum(r['decoy_chosen'] for r in report)}")
    if args.detect:
        print(f"  paper detector found the page: {sum(r['page_found'] for r in report)}/{n}")
    for key in ("difficulty", "header_mode"):
        by = defaultdict(Counter)
        for r in report:
            by[r[key]][r["outcome"]] += 1
        print(f"  by {key}: " + "  ".join(f"{k}: {c['right']}/{sum(c.values())} right, {c['WRONG']} wrong"
                                          for k, c in sorted(by.items())))
    for r in report:
        if r["outcome"] == "WRONG":
            print(f"  WRONG {r['id']}: {r['reason']} | got {r['name_got']} {r['ramq_got']} {r['dob_got']}"
                  f" | true {r['ramq_true']} | decoy={r['decoy_chosen']}")
    print(f"\nper-image report: {out}")


if __name__ == "__main__":
    main()
