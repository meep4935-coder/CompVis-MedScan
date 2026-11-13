// Background worker: loads the two reading models and the Python rules, then reads pictures.
// Running here keeps the page (and the camera preview) smooth while a read is in progress.

import { DEFAULTS, boxesFromMap, decodeLine, detSize, lineTensor, resizeNormBGR } from "./reader.js";

const BASE = new URL("./", import.meta.url).href;
const MODELS = { det: "models/PP-OCRv6_det_small.onnx", rec: "models/latin_PP-OCRv5_rec_mobile.onnx" };
const RULES = ["__init__", "ramq", "ctc", "extract", "vote"];

let ort, det, rec, chars, py, pyRead, pyCall;
const now = () => performance.now();
const say = (msg) => postMessage({ type: "progress", msg });

async function download(path) {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function loadModels(cfg, load) {
  let t = now();
  say("loading the model engine");
  const file = cfg.backend === "webgpu" ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs";
  ort = await import(`${BASE}vendor/ort/${file}`);
  ort.env.wasm.wasmPaths = `${BASE}vendor/ort/`;
  ort.env.wasm.numThreads = cfg.threads;
  load.engine = now() - t;

  t = now();
  say("downloading the two models");
  const [detBytes, recBytes, charList] = await Promise.all([
    download(MODELS.det), download(MODELS.rec), fetch(`${BASE}models/chars.json`).then((r) => r.json())]);
  chars = charList;
  load.download = now() - t;
  load.bytes = detBytes.length + recBytes.length;

  t = now();
  say("starting the models");
  const opts = { executionProviders: [cfg.backend], graphOptimizationLevel: "all" };
  det = await ort.InferenceSession.create(detBytes, opts);
  rec = await ort.InferenceSession.create(recBytes, opts);
  load.sessions = now() - t;
}

async function loadRules(load) {
  let t = now();
  const { loadPyodide } = await import(`${BASE}vendor/pyodide/pyodide.mjs`);
  py = await loadPyodide({ indexURL: `${BASE}vendor/pyodide/` });
  load.python = now() - t;

  t = now();
  py.FS.mkdirTree("/app/medscan");
  await Promise.all(RULES.map(async (name) => {
    const src = await fetch(`${BASE}py/medscan/${name}.py`).then((r) => r.text());
    py.FS.writeFile(`/app/medscan/${name}.py`, src);
  }));
  py.FS.writeFile("/app/rules.py", await fetch(`${BASE}rules.py`).then((r) => r.text()));
  py.runPython("import sys; sys.path.insert(0, '/app'); import rules");
  pyRead = py.globals.get("rules").read;
  pyCall = (name) => py.runPython(`import json; json.dumps(rules.${name}(), default=str)`);
  load.rules = now() - t;
}

async function init(cfg) {
  const load = { backend: cfg.backend, threads: cfg.threads };
  const t = now();
  // the models and Python load side by side
  await Promise.all([loadModels(cfg, load), loadRules(load)]);
  load.total = now() - t;
  postMessage({ type: "ready", load });
}

async function read({ id, buf, w, h, live, det: detOpts }) {
  const rgba = new Uint8ClampedArray(buf);
  const o = { ...DEFAULTS, ...detOpts };
  const time = {};
  let t = now();

  const { w1, h1, W, H } = detSize(w, h, o);
  const input = resizeNormBGR(rgba, w, h, W, H);
  time.resize = now() - t;

  t = now();
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, H, W]) });
  const map = out[det.outputNames[0]];
  time.find = now() - t;

  t = now();
  const [mh, mw] = map.dims.slice(-2);
  const boxes = boxesFromMap(map.data, mw, mh, w1, h1, w, h, o);
  time.boxes = now() - t;

  t = now();
  const items = [], lines = [];
  let tCut = 0;
  for (const b of boxes) {
    const t1 = now();
    const line = lineTensor(rgba, w, h, b.pts);
    tCut += now() - t1;
    if (!line) continue;
    const res = await rec.run({ [rec.inputNames[0]]: new ort.Tensor("float32", line.data, [1, 3, 48, line.width]) });
    const p = res[rec.outputNames[0]];
    const [T, C] = p.dims.slice(-2);
    const { text, conf, frames } = decodeLine(p.data, T, C, chars);
    if (!text.trim()) continue;
    const xs = b.pts.map((q) => q[0]), ys = b.pts.map((q) => q[1]);
    items.push([[Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], text, conf, frames]);
    lines.push({ pts: b.pts, text, conf });
  }
  time.lines = now() - t;
  time.cut = tCut;

  t = now();
  const result = JSON.parse(pyRead(JSON.stringify(items), live));
  time.rules = now() - t;
  time.total = time.resize + time.find + time.boxes + time.lines + time.rules;

  postMessage({ type: "result", id, w, h, detW: W, detH: H, nBoxes: boxes.length, lines, time, ...result });
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") await init(m);
    else if (m.type === "read") await read(m);
    else if (m.type === "reset" || m.type === "reject" || m.type === "accept") {
      postMessage({ type: m.type, value: JSON.parse(pyCall(m.type)) });
    }
  } catch (err) {
    postMessage({ type: "error", during: m.type, id: m.id, msg: String(err && err.message ? err.message : err) });
  }
};
