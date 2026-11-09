#!/usr/bin/env bash
# Phase 2: outline + track + capture (flattened scan) of paper on the MacBook camera.
#   bash mac/run_scan.sh                # paper only
#   bash mac/run_scan.sh --labels       # also labels on bottles / boxes
#   bash mac/run_scan.sh --conf 0.6     # any scan.py option can be appended
# Keys: q = quit, c = force-capture what's visible, s = save raw frame to captures/
# Scans are saved in captures/scans/<date_time>/
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scan.py --source 0 "$@"
