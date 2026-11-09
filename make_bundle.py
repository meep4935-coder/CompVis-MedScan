"""
Package the project for testing on another machine (e.g. the MacBook).

1. Copies the newest trained weights into models/ ("release" them):
       runs/paper_seg/weights/best.pt -> models/paper_seg.pt   (scan.py, phase 2)
2. Zips code + models/ into dist/CompV_bundle.zip (no dataset, no venv, no runs).

Usage:
    python make_bundle.py
    python make_bundle.py --weights runs/other/weights/best.pt --as paper_seg.pt
"""

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
INCLUDE = ["generate_synthetic.py", "train.py", "detect.py", "scan.py", "make_bundle.py",
           "requirements.txt", "README.md", ".gitignore", ".gitattributes",
           "mac/setup_mac.sh", "mac/run_webcam.sh", "mac/run_scan.sh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "paper_seg" / "weights" / "best.pt")
    ap.add_argument("--as", dest="as_name", default="paper_seg.pt", help="file name inside models/")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "CompV_bundle.zip")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights} (train first)")
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    shutil.copy2(args.weights, models / args.as_name)

    args.out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE + [f"models/{p.name}" for p in sorted(models.glob("*.pt"))]:
            p = ROOT / rel
            if not p.exists():
                print(f"  skip missing {rel}")
                continue
            info = zipfile.ZipInfo.from_file(p, f"CompV/{rel}")
            if rel.endswith(".sh"):
                info.external_attr = 0o100755 << 16  # regular file, executable on macOS
                data = p.read_bytes().replace(b"\r\n", b"\n")  # bash hates CRLF
            else:
                data = p.read_bytes()
            z.writestr(info, data, zipfile.ZIP_DEFLATED)
    print(f"Bundle: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
