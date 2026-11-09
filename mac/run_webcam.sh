#!/usr/bin/env bash
# Live paper detection + tracking on the MacBook camera.
#   bash mac/run_webcam.sh                 # defaults
#   bash mac/run_webcam.sh --conf 0.6      # any detect.py option can be appended
# Keys: q = quit, s = save raw frame to captures/
#
# First run: macOS will ask to give Terminal (or VS Code) camera access.
# If the window never shows video: System Settings > Privacy & Security > Camera,
# enable your terminal app, then run again.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python detect.py --source 0 "$@"
