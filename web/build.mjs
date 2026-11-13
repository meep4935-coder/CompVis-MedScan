// Assemble the web app into web/dist (not committed): the page, our Python rules, the two
// reading models and the two browser libraries. Run:  npm install  then  node build.mjs
//
// Nothing is copied by hand, so the page always uses the same medscan/*.py as the desktop.

import { copyFile, mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(web);
const dist = path.join(web, "dist");

const ORT = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
             "ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"];
const PYODIDE = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"];
const RULES = ["__init__.py", "ramq.py", "ctc.py", "extract.py", "vote.py"];
const MODELS = ["PP-OCRv6_det_small.onnx", "latin_PP-OCRv5_rec_mobile.onnx"];

/** RapidOCR downloads its models into its own package folder the first time the desktop reader runs. */
function modelDir() {
  const tried = [process.env.MEDSCAN_MODELS, path.join(repo, ".venv/Lib/site-packages/rapidocr/models")];
  const lib = path.join(repo, ".venv/lib");
  if (existsSync(lib)) {
    for (const d of ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10"]) {
      tried.push(path.join(lib, d, "site-packages/rapidocr/models"));
    }
  }
  const found = tried.find((d) => d && MODELS.every((m) => existsSync(path.join(d, m))));
  if (!found) {
    throw new Error("Reading models not found. Run the desktop reader once (python scan.py --read) so " +
                    "RapidOCR downloads them, or set MEDSCAN_MODELS to the folder holding:\n  " + MODELS.join("\n  "));
  }
  return found;
}

const sizes = {};
async function copy(from, to) {
  await mkdir(path.dirname(to), { recursive: true });
  await copyFile(from, to);
  sizes[path.relative(dist, to).replaceAll("\\", "/")] = (await stat(to)).size;
}

await rm(dist, { recursive: true, force: true });
for (const f of await readdir(path.join(web, "src"))) await copy(path.join(web, "src", f), path.join(dist, f));
for (const f of RULES) await copy(path.join(repo, "medscan", f), path.join(dist, "py/medscan", f));
for (const f of ORT) await copy(path.join(web, "node_modules/onnxruntime-web/dist", f), path.join(dist, "vendor/ort", f));
for (const f of PYODIDE) await copy(path.join(web, "node_modules/pyodide", f), path.join(dist, "vendor/pyodide", f));
const models = modelDir();
for (const f of MODELS) await copy(path.join(models, f), path.join(dist, "models", f));

// the recogniser's alphabet: index 0 is "nothing here" (blank), then the model's characters, then space
const dict = (await readFile(path.join(repo, "medscan/dicts/latin_PP-OCRv5_rec_mobile.txt"), "utf8")).split(/\r?\n/);
if (dict.at(-1) === "") dict.pop();
await writeFile(path.join(dist, "models/chars.json"), JSON.stringify(["", ...dict, " "]));

await writeFile(path.join(dist, ".nojekyll"), "");   // GitHub Pages: serve the files as they are
await writeFile(path.join(dist, "build.json"), JSON.stringify({ built: new Date().toISOString(), sizes }, null, 1));

const mb = (keys) => (keys.reduce((s, k) => s + sizes[k], 0) / 1e6).toFixed(1);
const all = Object.keys(sizes);
console.log(`built ${dist}`);
console.log(`  models   ${mb(all.filter((k) => k.startsWith("models/")))} MB`);
console.log(`  engine   ${mb(all.filter((k) => k.startsWith("vendor/ort/") && !k.includes("asyncify") && !k.includes("webgpu")))} MB` +
            ` (processor)  +${mb(all.filter((k) => k.includes("asyncify") || k.includes("webgpu")))} MB only if the graphics-chip engine is chosen`);
console.log(`  python   ${mb(all.filter((k) => k.startsWith("vendor/pyodide/")))} MB`);
console.log(`  page     ${mb(all.filter((k) => !k.startsWith("vendor/") && !k.startsWith("models/")))} MB`);
