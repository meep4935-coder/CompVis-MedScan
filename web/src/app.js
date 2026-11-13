// MedScan speed trial: camera / photo -> background worker -> what was read + how long it took.

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const cfg = { engine: q.get("engine") || "wasm", threads: q.get("threads") || "auto", size: q.get("size") || "full" };
const DET = { full: {}, 1280: { maxSide: 1280, minSide: 0 }, 960: { maxSide: 960, minSide: 0 } };
const ms = (v) => (v == null ? "" : v >= 100 ? Math.round(v) : v.toFixed(1));

const state = {
  ready: false, busy: false, live: false, stream: null, load: null,
  reads: [], lastGrab: 0, liveStart: 0, last: null, cam: "", proposal: null,
};
window.__trial = state;   // for looking at results from the browser's console
// for automated checks: await __trial.readUrl("sample_page.jpg") gives the full result
state.readUrl = (url) => new Promise((done) => {
  state.onResult = done;
  readStill(url);
});

// ------------------------------------------------------------------ start-up

// Several processor cores are only allowed on a "cross-origin isolated" page. The service
// worker (sw.js) adds the headers for that, which takes one reload the first time.
async function isolate() {
  if (q.get("coi") === "0" || self.crossOriginIsolated || !("serviceWorker" in navigator)) return;
  try {
    await navigator.serviceWorker.register("sw.js");
    await navigator.serviceWorker.ready;
    if (!sessionStorage.getItem("coi")) {
      sessionStorage.setItem("coi", "1");
      location.reload();
      await new Promise(() => {});
    }
  } catch (e) {
    console.warn("service worker unavailable", e);
  }
}

function threadCount() {
  if (!self.crossOriginIsolated) return 1;
  if (cfg.threads !== "auto") return Number(cfg.threads);
  return Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2)));
}

let worker;
async function start() {
  for (const k of ["engine", "threads", "size"]) {
    $(k).value = cfg[k];
    $(k).onchange = () => {
      q.set(k, $(k).value);
      location.search = q.toString();
    };
  }
  if (!navigator.gpu) $("engine").querySelector('[value="webgpu"]').disabled = true;
  await isolate();
  showDevice();
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = (e) => handlers[e.data.type]?.(e.data);
  worker.onerror = (e) => fail(`worker: ${e.message || "could not start"}`);
  worker.postMessage({ type: "init", backend: cfg.engine, threads: threadCount() });
}

function fail(msg) {
  $("error").textContent = msg;
  $("status").textContent = "Stopped.";
  state.busy = false;
}

const handlers = {
  progress: (m) => ($("status").textContent = `Loading: ${m.msg}…`),
  error: (m) => fail(`Error during ${m.during}: ${m.msg}` +
    (cfg.engine === "webgpu" ? "\nThe graphics-chip engine may not work here: switch Engine to Processor." : "")),
  ready(m) {
    state.ready = true;
    state.load = m.load;
    $("status").textContent = "Ready.";
    const L = m.load, rows = [
      ["Model engine", L.engine], [`Download the 2 models (${(L.bytes / 1e6).toFixed(1)} MB)`, L.download],
      ["Start the models", L.sessions], ["Python", L.python], ["Reading rules", L.rules]];
    $("loadTable").innerHTML = "<tr><th>Loading (side by side)</th><th class=n>ms</th></tr>" +
      rows.map(([a, b]) => `<tr><td>${a}</td><td class=n>${ms(b)}</td></tr>`).join("") +
      `<tr class=total><td>Ready after</td><td class=n>${ms(L.total)}</td></tr>`;
    $("loadTable").hidden = false;
    for (const id of ["camera", "photo", "samplePage", "sampleCard"]) $(id).disabled = false;
    showDevice();
    const auto = q.get("auto");
    if (auto) readStill(`sample_${auto}.jpg`);
  },
  result(m) {
    state.busy = false;
    m.time.grab = state.lastGrab;
    state.last = m;
    state.reads.push(m.time);
    drawBoxes(m);
    showRecord(m);
    showTimes(m);
    $("lines").innerHTML = m.lines.map((l) => `<li>${esc(l.text)} <span class=dim>${l.conf.toFixed(2)}</span></li>`).join("");
    state.onResult?.(m);
    state.onResult = null;
    if (!state.live) {
      $("status").textContent = m.sure ? "Photo read: sure enough to propose." :
        `Photo read, but not proposed (${m.why}). The app would ask for another photo.`;
      return;
    }
    if (m.proposal) {
      state.proposal = m.proposal;
      $("proposalText").innerHTML = recordHtml(m.proposal);
      $("proposalTime").textContent =
        `Proposed after ${((performance.now() - state.liveStart) / 1000).toFixed(1)} s and ${m.views} reads.`;
      $("proposal").hidden = false;
      $("status").textContent = "Waiting for accept or reject.";
    } else {
      $("status").textContent = m.status === "need_better_view" ? "Closer / more light please."
        : `Reading (${m.views} reads so far)…`;
      setTimeout(grabLive, 30);
    }
  },
  reject() {
    state.proposal = null;
    $("proposal").hidden = true;
    state.liveStart = performance.now();
    grabLive();
  },
  accept() {
    $("proposal").hidden = true;
    $("status").textContent = "Accepted. Press Start camera for the next patient.";
    stopCamera();
  },
  reset() {},
};

// ------------------------------------------------------------------ pictures in

function send(imageData, live) {
  state.busy = true;
  worker.postMessage({ type: "read", id: state.reads.length, buf: imageData.data.buffer, w: imageData.width,
                       h: imageData.height, live, det: DET[cfg.size] }, [imageData.data.buffer]);
}

const grabCanvas = document.createElement("canvas");
function grabLive() {
  const v = $("video");
  if (!state.live || state.busy || !v.videoWidth) return;
  const t = performance.now();
  grabCanvas.width = v.videoWidth;
  grabCanvas.height = v.videoHeight;
  const ctx = grabCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(v, 0, 0);
  const img = ctx.getImageData(0, 0, v.videoWidth, v.videoHeight);
  state.lastGrab = performance.now() - t;
  send(img, true);
}

async function startCamera() {
  if (state.live) return stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
  } catch (e) {
    return fail(`Camera: ${e.message}. Allow camera access for this page, or use a photo.`);
  }
  $("error").textContent = "";
  const v = $("video");
  v.srcObject = state.stream;
  await v.play();
  $("still").hidden = true;
  v.hidden = false;
  $("stage").style.display = "block";
  $("proposal").hidden = true;
  state.cam = `${v.videoWidth}x${v.videoHeight}`;
  state.live = true;
  state.liveStart = performance.now();
  $("camera").textContent = "Stop camera";
  $("status").textContent = "Reading…";
  showDevice();
  worker.postMessage({ type: "reset" });
  grabLive();
}

function stopCamera() {
  state.live = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("camera").textContent = "Start camera";
}

/** A photo or a built-in sample: read once, with no further views to vote with. */
async function readStill(src) {
  if (!state.ready || state.busy) return;
  stopCamera();
  $("proposal").hidden = true;
  $("status").textContent = "Reading the picture…";
  let img;
  try {
    const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
    img = await createImageBitmap(blob, { imageOrientation: "from-image" });   // turned the way it was taken
  } catch {
    return fail("That file could not be opened as a picture.");
  }
  const t = performance.now();
  const scale = Math.min(1, 3000 / Math.max(img.width, img.height));   // keep memory in check
  const c = $("still");
  c.width = Math.round(img.width * scale);
  c.height = Math.round(img.height * scale);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, c.width, c.height);
  img.close();
  $("video").hidden = true;
  c.hidden = false;
  $("stage").style.display = "block";
  const data = ctx.getImageData(0, 0, c.width, c.height);
  state.lastGrab = performance.now() - t;
  send(data, false);
}

// ------------------------------------------------------------------ showing results

const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function drawBoxes(m) {
  const o = $("overlay");
  o.width = m.w;
  o.height = m.h;
  const ctx = o.getContext("2d");
  ctx.lineWidth = Math.max(2, m.w / 500);
  ctx.strokeStyle = "#ff9d00";
  for (const l of m.lines) {
    ctx.beginPath();
    l.pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
  }
}

function recordHtml(r) {
  const ramq = r.ramq ? `${r.ramq.slice(0, 4)} ${r.ramq.slice(4, 8)} ${r.ramq.slice(8)}` : "not found";
  const name = r.last ? `${esc(r.last.toUpperCase())}, ${esc(r.first || "?")} ${esc(r.middle || "")}` : "not found";
  const dob = r.dob ? `${r.dob}${r.dob_source === "ramq" ? " (from NAM)" : ""}` : "not found";
  return `<div><b>Name</b> ${name}</div><div><b>RAMQ</b> ${ramq}</div><div><b>Born</b> ${dob}</div>` +
    (r.sex ? `<div><b>Sex</b> ${r.sex}</div>` : "");
}

function showRecord(m) {
  const r = m.reading, mark = (v) => (v ? '<span class=ok>agrees</span>' : '<span class=no>not confirmed</span>');
  $("record").className = "rec";
  $("record").innerHTML = recordHtml(r) +
    `<div class=dim>Name and RAMQ: ${mark(r.agree_name)} · Birth date and RAMQ: ` +
    (r.dob_source === "ramq" ? "no birth date was read; it was worked out from the RAMQ" : mark(r.agree_dob)) + "</div>";
}

const STEPS = [["grab", "Grab the picture"], ["resize", "Scale it for the text finder"], ["find", "Find text (model 1)"],
               ["boxes", "Outline the lines"], ["lines", "Read the lines (model 2)"], ["rules", "Rules (Python)"]];

function average(key, n = 10) {
  const r = state.reads.slice(-n);
  return r.reduce((s, t) => s + t[key], 0) / r.length;
}

function showTimes(m) {
  const t = m.time;
  $("timeTable").innerHTML = "<tr><th>Step</th><th class=n>this read, ms</th><th class=n>average of last 10</th></tr>" +
    STEPS.map(([k, name]) => `<tr><td>${name}</td><td class=n>${ms(t[k])}</td><td class=n>${ms(average(k))}</td></tr>`).join("") +
    `<tr class=total><td>Total</td><td class=n>${ms(t.total + t.grab)}</td><td class=n>${ms(average("total") + average("grab"))}</td></tr>`;
  $("readInfo").textContent = `${m.lines.length} lines read from ${m.nBoxes} boxes. Picture ${m.w}x${m.h}, ` +
    `text finder saw ${m.detW}x${m.detH}. ${state.reads.length} reads so far.`;
}

function deviceText() {
  const L = state.load;
  return [
    `Browser: ${navigator.userAgent}`,
    `Processor cores: ${navigator.hardwareConcurrency || "?"}; memory: ${navigator.deviceMemory ? navigator.deviceMemory + " GB" : "?"}`,
    `Engine: ${cfg.engine}; cores used: ${L ? L.threads : threadCount()}; isolated: ${self.crossOriginIsolated}; WebGPU: ${navigator.gpu ? "yes" : "no"}`,
    `Picture size setting: ${cfg.size}; camera: ${state.cam || "not started"}`,
  ].join("\n");
}
const showDevice = () => ($("device").textContent = deviceText());

function summary() {
  const L = state.load, out = ["MedScan speed trial", deviceText()];
  if (L) out.push(`Load ms: engine ${ms(L.engine)}, models download ${ms(L.download)}, start ${ms(L.sessions)}, ` +
    `python ${ms(L.python)}, rules ${ms(L.rules)}, total ${ms(L.total)}`);
  if (state.reads.length) {
    const m = state.last;
    out.push(`Reads: ${state.reads.length}; last picture ${m.w}x${m.h}, finder ${m.detW}x${m.detH}, ${m.lines.length} lines`);
    out.push("Average ms (last 10): " + STEPS.map(([k, n]) => `${k} ${ms(average(k))}`).join(", ") +
      `, total ${ms(average("total") + average("grab"))}`);
    if ($("proposalTime").textContent) out.push($("proposalTime").textContent);
  }
  return out.join("\n");
}

$("camera").onclick = startCamera;
$("photo").onclick = () => $("file").click();
$("file").onchange = (e) => e.target.files[0] && readStill(e.target.files[0]);
$("samplePage").onclick = () => readStill("sample_page.jpg");
$("sampleCard").onclick = () => readStill("sample_card.jpg");
$("accept").onclick = () => worker.postMessage({ type: "accept" });
$("reject").onclick = () => worker.postMessage({ type: "reject" });
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(summary());
    $("copied").textContent = "Copied.";
  } catch {
    $("copied").textContent = "Could not copy: select the text above instead.";
    $("device").textContent = summary();
  }
};

start();
