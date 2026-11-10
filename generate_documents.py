"""
Generate FAKE Quebec patient documents with exact ground truth.

Document types (all fictional patients; real Quebec hospital names as header text only):
    card          hospital card (carte d'hôpital): name, NAM, DDN, dossier, expiry
    sticker       patient ID sticker (étiquette) with barcode
    consult       hospital note / order sheet with a sticker + handwritten notes
    prescription  clinic prescription (ordonnance); patient info sometimes handwritten
    pharmacy      pharmacy bottle label: name (+ sometimes DDN), never the NAM
    handnote      handwritten sticky note / scrap of paper

For each document we save:
    data/docs/doc_00001.jpg
    data/docs/doc_00001.json   {"type", "patient": {...truth...},
                                "fields": [{"field": "name"|"last_name"|"given_names"|"ramq"|"dob",
                                            "text", "bbox": [x0,y0,x1,y1], "hand": bool}],
                                "lines":  [{"text", "bbox", "hand"}]}   # every text line, for OCR scoring

Distractors on purpose: record numbers, phone numbers, visit/expiry dates, licence numbers.

Usage:
    python generate_documents.py --n 300 --preview
"""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from medscan import qc_data as qc
from medscan.ramq import format_ramq

ROOT = Path(__file__).parent
FONT_DIRS = [ROOT / "fonts", Path("C:/Windows/Fonts"), Path("/Library/Fonts"),
             Path("/System/Library/Fonts/Supplemental"), Path("/usr/share/fonts/truetype")]
PRINT_FONTS = ["arial.ttf", "calibri.ttf", "times.ttf", "verdana.ttf", "tahoma.ttf", "georgia.ttf",
               "segoeui.ttf", "trebuc.ttf", "Arial.ttf", "Verdana.ttf", "Georgia.ttf"]
BOLD_FONTS = ["arialbd.ttf", "calibrib.ttf", "timesbd.ttf", "verdanab.ttf", "tahomabd.ttf", "segoeuib.ttf",
              "Arial Bold.ttf", "Verdana Bold.ttf"]
MONO_FONTS = ["consola.ttf", "cour.ttf", "courbd.ttf", "Courier New.ttf"]
HAND_FONTS = ["Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "mvboli.ttf", "comic.ttf", "Gabriola.ttf"]
INKS = [(25, 35, 140), (20, 20, 20), (10, 30, 90), (40, 40, 60)]

FILLER = ["Patient vu en clinique ce jour.", "Douleur thoracique atypique, ECG normal.",
          "Suivi dans 2 semaines.", "TA 132/84, FC 78, Sat 97 %", "Antécédents : HTA, DLP, DB2.",
          "Allergie : pénicilline", "Pas de changement de médication.", "Bilan sanguin (FSC, ions, créat).",
          "Plan : poursuivre traitement actuel.", "Rappeler la famille au 514-555-0199.",
          "Toux depuis 5 jours, afébrile.", "Revoir résultats d'imagerie.", "Référé en cardiologie."]
HAND_FILLER = ["rappeler famille", "RV 2 sem.", "TA 132/84", "voir labo", "renouv. Rx", "ECG ok",
               "appeler pharmacie", "f/u 3 mois", "allergie pénicilline", "chambre 412"]


def find_font(names):
    found = {}
    for n in names:
        for d in FONT_DIRS:
            p = d / n
            if p.exists():
                found.setdefault(str(p.resolve()).lower(), str(p))  # Windows paths are case-insensitive
                break
    return list(found.values())


FONTS = {k: find_font(v) for k, v in
         {"print": PRINT_FONTS, "bold": BOLD_FONTS, "mono": MONO_FONTS, "hand": HAND_FONTS}.items()}
_cache = {}


def font(kind, size, path=None):
    path = path or random.choice(FONTS[kind])
    key = (path, int(size))
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, int(size))
    return _cache[key]


# ----------------------------------------------------------------------------
# A drawable document that remembers where every piece of text went
# ----------------------------------------------------------------------------
class Doc:
    def __init__(self, w, h, bg=(250, 250, 248)):
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img)
        self.fields, self.lines = [], []

    def text(self, xy, s, f, fill=(20, 20, 20), field=None, hand=False):
        """Draw text, record its box. Returns the box (x0, y0, x1, y1)."""
        bbox = self._hand(xy, s, f, fill) if hand else self._print(xy, s, f, fill)
        self.lines.append({"text": s, "bbox": bbox, "hand": hand})
        if field:
            self.fields.append({"field": field, "text": s, "bbox": bbox, "hand": hand})
        return bbox

    def label_value(self, x, y, label, value, f_label, f_value, field, hand=False, fill=(20, 20, 20)):
        """'NAM : TREM 8555 1512' where only the value gets the field box."""
        b = self.text((x, y), label, f_label, fill=fill)
        gap = random.randint(6, 20)
        b2 = self.text((b[2] + gap, y), value, f_value, fill=fill, field=field, hand=hand)
        return (b[0], min(b[1], b2[1]), b2[2], max(b[3], b2[3]))

    def _print(self, xy, s, f, fill):
        self.d.text(xy, s, font=f, fill=fill)
        return list(self.d.textbbox(xy, s, font=f))

    def _hand(self, xy, s, f, fill):
        """Handwriting look: every word jittered in size, height and angle; line drifts."""
        x, y = xy
        slope = random.uniform(-0.04, 0.04)
        x0 = x
        boxes = []
        size = f.size
        for word in s.split(" "):
            if not word:
                x += size * 0.3
                continue
            wf = font("hand", size * random.uniform(0.92, 1.08), f.path)
            bb = wf.getbbox(word)
            pad = 8
            w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
            tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((pad - bb[0], pad - bb[1]), word, font=wf, fill=fill + (255,))
            tile = tile.rotate(random.uniform(-4, 4), expand=True, resample=Image.BICUBIC)
            px = int(x)
            py = int(y + (x - x0) * slope + random.uniform(-2, 2) + bb[1] - pad)
            self.img.paste(tile, (px, py), tile)
            tb = tile.getbbox()
            if tb:
                boxes.append((px + tb[0], py + tb[1], px + tb[2], py + tb[3]))
            x += (bb[2] - bb[0]) + wf.getlength(" ") * random.uniform(0.8, 1.5)
        if not boxes:
            return [int(xy[0]), int(xy[1]), int(xy[0]) + 1, int(xy[1]) + 1]
        return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]

    def paste(self, other, xy):
        """Paste another Doc (e.g. a sticker) and carry its boxes along."""
        ox, oy = xy
        self.img.paste(other.img, (ox, oy))
        shift = lambda b: [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
        for L in other.lines:
            self.lines.append({**L, "bbox": shift(L["bbox"])})
        for F in other.fields:
            self.fields.append({**F, "bbox": shift(F["bbox"])})


def barcode(doc, x, y, w, h):
    while w > 4:
        bw = random.randint(1, 4)
        doc.d.rectangle([x, y, x + bw - 1, y + h], fill=(0, 0, 0))
        step = bw + random.randint(1, 4)
        x, w = x + step, w - step


def given(p):
    return " ".join(x for x in (p.first, p.middle) if x)


# ----------------------------------------------------------------------------
# Document types
# ----------------------------------------------------------------------------
def make_card(p, hosp):
    W, H = 1012, 638
    bg = random.choice([(250, 250, 250), (235, 242, 250), (250, 245, 230), (240, 250, 240)])
    doc = Doc(W, H, bg)
    band = random.choice([(0, 70, 140), (150, 20, 40), (0, 110, 90), (60, 60, 60), (100, 40, 120)])
    doc.d.rectangle([0, 0, W, 150], fill=band)
    size = 40
    fb = font("bold", size)
    while fb.getlength(hosp[0]) > W - 80 and size > 16:  # shrink long hospital names to fit
        size -= 2
        fb = font("bold", size, fb.path)
    doc.text((40, 30), hosp[0], fb, fill=(255, 255, 255))
    doc.text((40, 95), random.choice(["Carte d'hôpital", "CARTE D'IDENTIFICATION", "Carte patient"]),
             font("print", 30), fill=(230, 230, 230))
    fl, fv = font("print", 26), font(random.choice(["mono", "bold"]), 40)
    ink = (20, 20, 20) if random.random() < 0.6 else (60, 60, 70)
    y = 185
    if random.random() < 0.5:
        doc.label_value(40, y, "NOM", p.last.upper(), fl, fv, "last_name", fill=ink)
        y += 62
        doc.label_value(40, y, "PRÉNOM", given(p).upper(), fl, fv, "given_names", fill=ink)
    else:
        doc.text((40, y), f"{p.last.upper()}, {given(p).upper()}", fv, fill=ink, field="name")
    y += 72
    doc.label_value(40, y, random.choice(["NAM", "No ass. maladie", "RAMQ"]), format_ramq(p.ramq, random.choice(["spaced", "compact"])), fl, fv, "ramq", fill=ink)
    y += 72
    doc.label_value(40, y, random.choice(["DDN", "Naissance", "Date de naissance"]), qc.format_date(p.dob), fl, fv, "dob", fill=ink)
    doc.label_value(620, y, "Sexe", p.sex, fl, fv, None, fill=ink)
    y += 72
    doc.label_value(40, y, random.choice(["DOSSIER", "No dossier", "MRN"]), p.dossier, fl, fv, None, fill=ink)
    exp = date(2026, 1, 1) + timedelta(days=random.randint(0, 1500))
    doc.label_value(620, y, "EXP", exp.strftime("%Y-%m"), fl, fv, None, fill=ink)
    return doc


def make_sticker(p, hosp, scale=1.0):
    W, H = int(640 * scale), int(270 * scale)
    doc = Doc(W, H, (255, 255, 255))
    doc.d.rectangle([0, 0, W - 1, H - 1], outline=(200, 200, 200), width=2)
    f = font("mono" if random.random() < 0.5 else "print", 30 * scale)
    fb = font("bold", 32 * scale)
    lh = int(42 * scale)
    x, y = int(16 * scale), int(12 * scale)
    doc.text((x, y), qc.format_name(p), fb, field="name")
    y += lh
    b = doc.label_value(x, y, "NAM:", format_ramq(p.ramq, random.choice(["spaced", "compact"])), f, f, "ramq")
    doc.text((b[2] + 30, y), f"Sexe: {p.sex}", f)
    y += lh
    age = 2026 - p.dob.year
    b = doc.label_value(x, y, random.choice(["DDN:", "Né(e):", "D.N.:"]), qc.format_date(p.dob), f, f, "dob")
    doc.text((b[2] + 20, y), f"({age} ans)", f)
    y += lh
    doc.text((x, y), f"Dossier: {p.dossier}   {hosp[1]}", f)
    y += lh
    barcode(doc, x, y + 4, int(W * 0.6), int(40 * scale))
    return doc


def make_consult(p, hosp):
    W, H = 1240, 1754
    doc = Doc(W, H, (252, 252, 250))
    fb, f = font("bold", 34), font("print", 26)
    doc.text((60, 60), hosp[0], fb)
    doc.text((60, 110), f"{hosp[2]} (Québec)", f)
    doc.text((60, 200), random.choice(["NOTE DE CONSULTATION", "FEUILLE D'ORDONNANCES", "NOTE D'ÉVOLUTION",
                                       "DEMANDE DE CONSULTATION"]), font("bold", 40))
    st = make_sticker(p, hosp, scale=random.uniform(0.85, 1.0))
    doc.paste(st, (W - st.img.width - 50, 40))
    visit = date(2026, 1, 1) + timedelta(days=random.randint(0, 330))
    doc.text((60, 300), f"Date : {visit.isoformat()}    Heure : {random.randint(7, 19)}:{random.choice(['00', '15', '30', '45'])}", f)
    y = 380
    for i in range(random.randint(14, 24)):  # ruled lines with mixed printed / handwritten notes
        doc.d.line([60, y + 40, W - 60, y + 40], fill=(200, 200, 215), width=1)
        r = random.random()
        if r < 0.45:
            doc.text((70, y), random.choice(HAND_FILLER + FILLER), font("hand", random.randint(28, 36)),
                     fill=random.choice(INKS), hand=True)
        elif r < 0.65:
            doc.text((70, y + 6), random.choice(FILLER), f)
        y += 54
        if y > H - 250:
            break
    doc_name, lic = qc.physician()
    doc.text((60, H - 180), f"{doc_name}, permis {lic}", f)
    doc.text((60, H - 130), doc_name.split()[-1], font("hand", 44), fill=random.choice(INKS), hand=True)
    return doc


def make_prescription(p, _hosp):
    W, H = 1240, 1300
    doc = Doc(W, H, random.choice([(252, 252, 250), (245, 248, 255)]))
    clinic = random.choice(qc.CLINICS)
    fb, f = font("bold", 38), font("print", 26)
    doc.text((60, 50), clinic, fb)
    doc.text((60, 100), f"{random.randint(10, 9999)}, {random.choice(qc.STREETS)}, {random.choice(qc.HOSPITALS)[2]}  "
                        f"Tél. : {random.choice(['514', '450', '418'])}-555-{random.randint(0, 9999):04d}", f)
    doc.text((60, 170), "ORDONNANCE", font("bold", 44))
    hand = random.random() < 0.45  # physician fills patient info by hand
    fv = font("hand", random.randint(34, 42)) if hand else font("print", 30)
    ink = random.choice(INKS) if hand else (20, 20, 20)
    y = 260
    doc.label_value(60, y, "Patient :", qc.format_name(p), f, fv, "name", hand=hand, fill=ink if hand else (20, 20, 20))
    y += 70
    doc.label_value(60, y, random.choice(["NAM :", "No RAMQ :", "Ass. maladie :"]), format_ramq(p.ramq, random.choice(["spaced", "compact", "dash"])),
                    f, fv, "ramq", hand=hand, fill=ink if hand else (20, 20, 20))
    y += 70
    doc.label_value(60, y, random.choice(["DDN :", "Date de naissance :"]), qc.format_date(p.dob), f, fv, "dob",
                    hand=hand, fill=ink if hand else (20, 20, 20))
    visit = date(2026, 1, 1) + timedelta(days=random.randint(0, 330))
    doc.text((780, 260), f"Date : {visit.isoformat()}", f)
    y += 110
    doc.text((60, y), "Rx", font("bold", 60))
    y += 90
    for med, dose, sig in random.sample(qc.MEDS, random.randint(1, 3)):
        line = f"{med} {dose} — {sig} — Qté : {random.choice([30, 60, 90])} — Ren. : {random.randint(0, 5)}"
        if hand:
            doc.text((90, y), f"{med} {dose}  {sig}", font("hand", 34), fill=ink, hand=True)
        else:
            doc.text((90, y), line, f)
        y += 70
    doc_name, lic = qc.physician()
    doc.text((60, H - 170), f"{doc_name}   Permis : {lic}", f)
    doc.text((60, H - 120), doc_name.split()[-1], font("hand", 46), fill=random.choice(INKS), hand=True)
    return doc


def make_pharmacy(p, _hosp):
    W, H = 1000, 430
    doc = Doc(W, H, random.choice([(255, 255, 255), (255, 252, 235)]))
    accent = random.choice([(0, 90, 160), (200, 30, 40), (0, 120, 70)])
    doc.d.rectangle([0, 0, W, 70], fill=accent)
    doc.text((20, 14), random.choice(qc.PHARMACIES), font("bold", 36), fill=(255, 255, 255))
    f, fs = font(random.choice(["print", "mono"]), 26), font("print", 22)
    doc.text((20, 82), f"{random.randint(10, 9999)}, {random.choice(qc.STREETS)}  Tél. : 450-555-{random.randint(0, 9999):04d}", fs)
    filled = date(2026, 1, 1) + timedelta(days=random.randint(0, 330))
    doc.text((20, 120), f"Rx : {random.randint(1000000, 9999999)}     Date : {filled.isoformat()}", f)
    doc.text((20, 160), qc.format_name(p), font("bold", 36), field="name")
    y = 210
    if random.random() < 0.25:
        doc.label_value(20, y, "DDN :", qc.format_date(p.dob), f, f, "dob")
        y += 40
    med, dose, sig = random.choice(qc.MEDS)
    doc.text((20, y), f"{med.upper()} {dose}", font("bold", 30))
    doc.text((20, y + 45), sig.capitalize(), f)
    doc_name, _ = qc.physician()
    doc.text((20, y + 90), f"Qté : {random.choice([30, 60, 90])}   Ren. : {random.randint(0, 5)}   {doc_name}", fs)
    return doc


def make_handnote(p, _hosp):
    sticky = random.random() < 0.5
    W, H = (650, 650) if sticky else (1000, 720)
    bg = random.choice([(255, 240, 140), (255, 210, 225), (200, 240, 255), (210, 250, 200)]) if sticky else (250, 250, 245)
    doc = Doc(W, H, bg)
    if not sticky:
        for yy in range(90, H, 56):
            doc.d.line([0, yy, W, yy], fill=(170, 190, 230), width=1)
    hf_path = random.choice(FONTS["hand"])
    hf = font("hand", random.randint(36, 46), hf_path)
    ink = random.choice(INKS)
    pre = "" if random.random() < 0.6 else ("Mme " if p.sex == "F" else "M. ")
    name = random.choice([p.full, f"{p.first} {p.last}", f"{p.last}, {given(p)}", p.full.upper()])
    lines = [("name", pre, name),
             ("ramq", random.choice(["NAM ", "nam: ", "RAMQ ", ""]), format_ramq(p.ramq, random.choice(["spaced", "compact"]))),
             ("dob", random.choice(["DDN ", "né le " if p.sex == "M" else "née le ", "ddn: ", ""]), qc.format_date(p.dob))]
    if random.random() < 0.3:
        lines.pop(random.randrange(3))  # not every note has all three fields
    lines += [(None, "", random.choice(HAND_FILLER)) for _ in range(random.randint(0, 2))]
    random.shuffle(lines)
    y = random.randint(40, 80)
    for fieldname, label, value in lines:
        x = random.randint(25, 60)
        if label:
            b = doc.text((x, y), label.strip(), hf, fill=ink, hand=True)
            x = b[2] + random.randint(10, 22)
        doc.text((x, y), value, hf, fill=ink, field=fieldname, hand=True)
        y += random.randint(70, 95)
        if y > H - 60:
            break
    return doc


MAKERS = {"card": make_card, "sticker": make_sticker, "consult": make_consult,
          "prescription": make_prescription, "pharmacy": make_pharmacy, "handnote": make_handnote}
WEIGHTS = {"card": 2, "sticker": 2, "consult": 2, "prescription": 2, "pharmacy": 2, "handnote": 3}


def finish(doc):
    """Light print/scan wear: slight blur, ink fade. Camera effects come later (scenes)."""
    img = doc.img
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "docs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", default=",".join(MAKERS), help="comma list of document types")
    ap.add_argument("--preview", action="store_true", help="also write images with field boxes drawn")
    args = ap.parse_args()

    random.seed(args.seed)
    if not FONTS["hand"] or not FONTS["print"]:
        raise SystemExit("No fonts found; put .ttf files in fonts/")
    print(f"fonts: {len(FONTS['print'])} print, {len(FONTS['bold'])} bold, {len(FONTS['hand'])} handwriting")
    types = args.types.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    colors = {"name": "green", "last_name": "green", "given_names": "lime", "ramq": "red", "dob": "blue"}
    for i in range(args.n):
        t = random.choices(types, weights=[WEIGHTS[k] for k in types])[0]
        p = qc.random_patient()
        hosp = random.choice(qc.HOSPITALS)
        doc = MAKERS[t](p, hosp)
        img = finish(doc)
        stem = f"doc_{i:05d}"
        img.save(args.out / f"{stem}.jpg", quality=92)
        truth = {"first": p.first, "middle": p.middle, "last": p.last, "sex": p.sex,
                 "dob": p.dob.isoformat(), "ramq": p.ramq, "dossier": p.dossier}
        meta = {"type": t, "hospital": hosp[0] if t in ("card", "sticker", "consult") else None,
                "size": img.size, "patient": truth, "fields": doc.fields, "lines": doc.lines}
        (args.out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.preview and i < 30:
            pv = img.copy()
            dr = ImageDraw.Draw(pv)
            for F in doc.fields:
                dr.rectangle(F["bbox"], outline=colors[F["field"]], width=3)
            (args.out / "preview").mkdir(exist_ok=True)
            pv.save(args.out / "preview" / f"{stem}_{t}.jpg", quality=85)
    print(f"Done: {args.n} documents in {args.out}")


if __name__ == "__main__":
    main()
