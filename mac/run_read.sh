#!/usr/bin/env bash
# Phase 3: read the patient's name, RAMQ and DOB from what the MacBook camera sees.
#   bash mac/run_read.sh                # paper only
#   bash mac/run_read.sh --labels       # also labels on bottles / boxes
# Hold the document still-ish and fairly close (text should be readable on screen).
# When a record is proposed at the bottom: Enter = accept, Backspace or r = reject.
# Keys: q = quit. Accepted records: captures/scans/<date_time>/patient_N.json (never committed).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
if ! python -c "import rapidocr" 2>/dev/null; then
    pip install "rapidocr>=3.0"
    # optional: runs the reading models a bit faster; not available on every Mac / Python
    pip install onnxruntime 2>/dev/null || echo "onnxruntime not available here: OpenCV will run the models (same results)"
fi
python scan.py --source 0 --read "$@"
