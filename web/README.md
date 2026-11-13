# MedScan in the browser: speed trial

A bare test page for the first question of the web app (see
[../docs/web-app-design.md](../docs/web-app-design.md)): **is a phone fast enough to
do the reading itself, with nothing uploaded?**

The page opens the camera (or takes a photo, or uses a built-in fictional sample),
reads the name, RAMQ and date of birth, and shows how long each step took.

## How it works

| Step | Runs as | File |
|---|---|---|
| Camera, photo, timings | JavaScript in the page | `src/app.js` |
| Find the text lines (model 1), read each line (model 2) | ONNX Runtime Web, in a background worker | `src/worker.js`, `src/reader.js` |
| Name / RAMQ / date rules, voting across views | **The desktop's own Python files**, run by Pyodide | `../medscan/*.py`, `src/rules.py` |
| Refuse any request to another site; allow several processor cores | Service worker | `src/sw.js` |

`src/reader.js` repeats in plain JavaScript what RapidOCR does on the desktop between
the two models (scale the picture, turn the detector's map into boxes, cut out and
straighten each line, decode the output). The models are the same files.

The rules are not copied: the build takes `medscan/ramq.py`, `ctc.py`, `extract.py`
and `vote.py` as they are, so a fix on the desktop reaches the page at the next build.

## Build and try it on this computer

```bash
cd web
npm install        # once: ONNX Runtime Web and Pyodide
node build.mjs     # assembles web/dist (not committed)
node serve.mjs     # http://localhost:5173
```

The build needs the two reading models, which RapidOCR downloads the first time the
desktop reader runs (`python scan.py --read`).

A phone cannot use the local server for the camera (browsers only allow the camera on
https pages). `node deploy.mjs` publishes `web/dist` as a public GitHub Pages site.

## What to look at

- **Ready after**: the first load (download plus start-up). Later visits use the
  browser's cache.
- **Time per read**: the total, and which step dominates.
- **Proposed after N s**: with the camera, how long until a patient is proposed
  (3 agreeing reads are needed).
- Try the three settings: engine (processor or graphics chip), processor cores, and
  the picture size for finding text.
- **Copy timings** copies device details and timings only, never what was read.

Use fictional documents only on this page.

## Checked on the desktop (built-in Chromium browser, 4 cores)

Same reader, same 40 camera-like fictional pages, one photo each:

| | Right | Skipped (not sure) | Wrong |
|---|---|---|---|
| Desktop (Python, onnxruntime) | 21 | 17 | 2 |
| Browser (this page) | 22 | 16 | 2 |

The two give the same patient record on 38 of the 40 pages. Where single letters
differ, the line is too blurry for either to read. (These single blurred photos are
deliberately hard; with the camera, several views are voted.)

Time per read on this desktop, sample page (1240x1754, 22 lines) / sample card:

| Setting | Page | Card |
|---|---|---|
| Processor, 4 cores | 1.33 s | 0.65 s |
| Processor, 1 core | 3.65 s | 1.70 s |
| Processor, 4 cores, picture 960 px | 0.68 s | 0.51 s |
| Graphics chip (WebGPU) | 0.55 s | 0.34 s |
| Desktop Python, for comparison | 0.64 s | 0.33 s |

First load: 46 MB of files (models 17.8, model engine 14.9, Python 13.5), about
26 to 30 MB over the network once the host compresses them; ready in about 1.5 s
from a local server. The graphics-chip engine is a further 29 MB, loaded only if chosen.

Not yet measured: any phone, Safari, or a real camera.

## Licences of what the build bundles

ONNX Runtime Web (MIT), Pyodide (MPL-2.0), PaddleOCR PP-OCR models via RapidOCR
(Apache-2.0).
