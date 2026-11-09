#!/usr/bin/env bash
# One-time setup on macOS. Run from the project folder:  bash mac/setup_mac.sh
set -e
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null; then
  echo "python3 not found. Install it from https://www.python.org/downloads/macos/ (or: brew install python)"
  exit 1
fi
echo "Using $(python3 --version)"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # pulls torch automatically (Apple Silicon build included)

python -c "import torch, ultralytics, cv2; print('torch', torch.__version__, '| MPS (Apple GPU):', torch.backends.mps.is_available()); print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)"
echo
echo "Setup done. Start the webcam demo with:  bash mac/run_webcam.sh"
