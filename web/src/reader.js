// Text reader for the browser: the same steps as RapidOCR on the desktop (medscan/ocr.py).
//
//   1. detSize / resizeNormBGR   scale the picture for the text detector
//   2. boxesFromMap              turn the detector's "text here" map into line boxes
//   3. lineTensor                cut each line out, straighten it, scale it to 48 px high
//   4. decodeLine                read the recogniser's output, keeping the runners-up
//
// Pictures are RGBA bytes (canvas ImageData). The models were trained on BGR, so
// channels are swapped here. Everything is plain JavaScript: no OpenCV needed.

export const DEFAULTS = {
  maxSide: 2000,   // pictures longer than this are shrunk for the detector
  minSide: 736,    // pictures shorter than this are enlarged for the detector (0 = never)
  thresh: 0.3,     // a map pixel counts as text above this
  boxThresh: 0.5,  // a box is kept if its average map value is above this
  unclip: 1.6,     // how much each box is grown (the map marks only the core of a line)
  minSize: 3,
  maxBoxes: 1000,
};

// Python's round(): halves go to the even number (so sizes match the desktop exactly).
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
const to32 = (v) => pyRound(Math.trunc(v) / 32) * 32;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Size the detector sees. w1 x h1: after the "too big" rule; W x H: what goes into the model. */
export function detSize(w, h, o = DEFAULTS) {
  let w1 = w, h1 = h;
  if (Math.max(w, h) > o.maxSide) {
    const r = o.maxSide / Math.max(w, h);
    w1 = to32(w * r);
    h1 = to32(h * r);
  }
  let r2 = 1;
  if (o.minSide && Math.min(w1, h1) < o.minSide) r2 = o.minSide / Math.min(w1, h1);
  return { w1, h1, W: Math.max(32, to32(w1 * r2)), H: Math.max(32, to32(h1 * r2)) };
}

/** RGBA picture -> Float32 [3, H, W] in BGR order, scaled to -1..1 (bilinear, like cv2.resize). */
export function resizeNormBGR(rgba, w, h, W, H) {
  const plane = W * H, out = new Float32Array(3 * plane);
  const xa = new Int32Array(W), xb = new Int32Array(W), xf = new Float32Array(W);
  const sx = w / W, sy = h / H;
  for (let i = 0; i < W; i++) {
    const s = clamp((i + 0.5) * sx - 0.5, 0, w - 1), a = Math.floor(s);
    xa[i] = a * 4;
    xb[i] = Math.min(a + 1, w - 1) * 4;
    xf[i] = s - a;
  }
  for (let j = 0; j < H; j++) {
    const s = clamp((j + 0.5) * sy - 0.5, 0, h - 1), a = Math.floor(s), fy = s - a;
    const ra = a * w * 4, rb = Math.min(a + 1, h - 1) * w * 4;
    let o = j * W;
    for (let i = 0; i < W; i++, o++) {
      const fx = xf[i], p00 = ra + xa[i], p01 = ra + xb[i], p10 = rb + xa[i], p11 = rb + xb[i];
      const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
      out[o] = (rgba[p00 + 2] * w00 + rgba[p01 + 2] * w01 + rgba[p10 + 2] * w10 + rgba[p11 + 2] * w11) / 127.5 - 1;
      out[plane + o] = (rgba[p00 + 1] * w00 + rgba[p01 + 1] * w01 + rgba[p10 + 1] * w10 + rgba[p11 + 1] * w11) / 127.5 - 1;
      out[2 * plane + o] = (rgba[p00] * w00 + rgba[p01] * w01 + rgba[p10] * w10 + rgba[p11] * w11) / 127.5 - 1;
    }
  }
  return out;
}

// ---------------------------------------------------------------- boxes from the map

function convexHull(pts) {
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Smallest rotated rectangle around the points: one side always lies along a hull edge. */
function minAreaRect(hull) {
  if (hull.length < 3) return null;
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len === 0) continue;
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy, v = -p[0] * uy + p[1] * ux;
      if (u < u0) u0 = u;
      if (u > u1) u1 = u;
      if (v < v0) v0 = v;
      if (v > v1) v1 = v;
    }
    const area = (u1 - u0) * (v1 - v0);
    if (!best || area < best.area) {
      const cu = (u0 + u1) / 2, cv = (v0 + v1) / 2;
      best = { area, ux, uy, w: u1 - u0, h: v1 - v0, cx: cu * ux - cv * uy, cy: cu * uy + cv * ux };
    }
  }
  return best;
}

function corners(r, w = r.w, h = r.h) {
  const ax = (r.ux * w) / 2, ay = (r.uy * w) / 2, bx = (-r.uy * h) / 2, by = (r.ux * h) / 2;
  return [[r.cx - ax - bx, r.cy - ay - by], [r.cx + ax - bx, r.cy + ay - by],
          [r.cx + ax + bx, r.cy + ay + by], [r.cx - ax + bx, r.cy - ay + by]];
}

/** Order 4 corners as top-left, top-right, bottom-right, bottom-left. */
function orderCorners(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0]);
  const [tl, bl] = p[1][1] > p[0][1] ? [p[0], p[1]] : [p[1], p[0]];
  const [tr, br] = p[3][1] > p[2][1] ? [p[2], p[3]] : [p[3], p[2]];
  return [tl, tr, br, bl];
}

/** Average map value inside the rotated rectangle. */
function boxScore(pred, mw, mh, r) {
  const c = corners(r);
  const x0 = clamp(Math.floor(Math.min(...c.map((p) => p[0]))), 0, mw - 1);
  const x1 = clamp(Math.ceil(Math.max(...c.map((p) => p[0]))), 0, mw - 1);
  const y0 = clamp(Math.floor(Math.min(...c.map((p) => p[1]))), 0, mh - 1);
  const y1 = clamp(Math.ceil(Math.max(...c.map((p) => p[1]))), 0, mh - 1);
  const hw = r.w / 2 + 0.5, hh = r.h / 2 + 0.5;
  let sum = 0, n = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - r.cx, dy = y - r.cy;
      if (Math.abs(dx * r.ux + dy * r.uy) <= hw && Math.abs(-dx * r.uy + dy * r.ux) <= hh) {
        sum += pred[y * mw + x];
        n++;
      }
    }
  }
  return n ? sum / n : 0;
}

/**
 * The detector's map (mw x mh, values 0..1) -> text-line boxes in the ORIGINAL picture
 * (w x h). Each box: 4 corners [tl, tr, br, bl]. Sorted top to bottom, then left to right.
 */
export function boxesFromMap(pred, mw, mh, w1, h1, w, h, o = DEFAULTS) {
  const n = mw * mh, mask = new Uint8Array(n);
  for (let y = 0, i = 0; y < mh; y++) {       // threshold + grow by one pixel (2x2 dilation)
    for (let x = 0; x < mw; x++, i++) {
      const t = o.thresh;
      mask[i] = pred[i] > t || (x > 0 && pred[i - 1] > t) || (y > 0 && pred[i - mw] > t) ||
                (x > 0 && y > 0 && pred[i - mw - 1] > t) ? 1 : 0;
    }
  }
  const seen = new Uint8Array(n), stack = new Int32Array(n);
  const rowMin = new Int32Array(mh).fill(mw), rowMax = new Int32Array(mh).fill(-1);
  const boxes = [];
  let count = 0;
  for (let start = 0; start < n && count < o.maxBoxes; start++) {
    if (!mask[start] || seen[start]) continue;
    count++;
    let top = 0, ymin = mh, ymax = -1;    // flood-fill one connected blob (8 neighbours)
    stack[top++] = start;
    seen[start] = 1;
    while (top) {
      const i = stack[--top], x = i % mw, y = (i - x) / mw;
      if (x < rowMin[y]) rowMin[y] = x;
      if (x > rowMax[y]) rowMax[y] = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= mh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= mw) continue;
          const j = yy * mw + xx;
          if (mask[j] && !seen[j]) {
            seen[j] = 1;
            stack[top++] = j;
          }
        }
      }
    }
    const pts = [];                        // the blob's left and right ends on each row
    for (let y = ymin; y <= ymax; y++) {
      pts.push([rowMin[y], y]);
      if (rowMax[y] !== rowMin[y]) pts.push([rowMax[y], y]);
      rowMin[y] = mw;
      rowMax[y] = -1;
    }
    const r = minAreaRect(convexHull(pts));
    if (!r || Math.min(r.w, r.h) < o.minSize) continue;
    const score = boxScore(pred, mw, mh, r);
    if (score < o.boxThresh) continue;
    const d = (r.w * r.h * o.unclip) / (2 * (r.w + r.h));   // grow the box on every side
    // the desktop's polygon library works on whole pixels (it drops the fractions first)
    const whole = minAreaRect(convexHull(corners(r).map(([x, y]) => [Math.trunc(x), Math.trunc(y)])));
    if (!whole) continue;
    const gw = whole.w + 2 * d, gh = whole.h + 2 * d;
    if (Math.min(gw, gh) < o.minSize + 2) continue;
    let b = orderCorners(corners(whole, gw, gh)).map(([x, y]) => [
      clamp(Math.round((x / mw) * w1), 0, w1), clamp(Math.round((y / mh) * h1), 0, h1)]);
    b = orderCorners(b).map(([x, y]) => [clamp(x, 0, w1 - 1), clamp(y, 0, h1 - 1)]);
    const bw = Math.trunc(Math.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]));
    const bh = Math.trunc(Math.hypot(b[0][0] - b[3][0], b[0][1] - b[3][1]));
    if (bw <= 3 || bh <= 3) continue;
    boxes.push({ pts: b, score });
  }
  // reading order: rows top to bottom (boxes within 10 px count as one row), then left to right
  boxes.sort((a, b) => a.pts[0][1] - b.pts[0][1]);
  let line = 0;
  boxes.forEach((b, i) => {
    if (i && b.pts[0][1] - boxes[i - 1].pts[0][1] >= 10) line++;
    b.line = line;
  });
  boxes.sort((a, b) => a.line - b.line || a.pts[0][0] - b.pts[0][0]);
  const rx = w / w1, ry = h / h1;
  return boxes.map((b) => ({
    score: b.score,
    pts: b.pts.map(([x, y]) => [clamp(x * rx, 0, w), clamp(y * ry, 0, h)]),
  }));
}

// ---------------------------------------------------------------- one text line

/**
 * Cut one line out of the picture, straighten it and scale it to 48 px high for the
 * recogniser. Returns { data: Float32 [3, 48, width], width } or null if the box is tiny.
 * A tall narrow box is taken as sideways text and turned 90 degrees, as on the desktop.
 */
export function lineTensor(rgba, w, h, pts) {
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const cw = Math.trunc(Math.max(dist(pts[0], pts[1]), dist(pts[2], pts[3])));
  const ch = Math.trunc(Math.max(dist(pts[0], pts[3]), dist(pts[1], pts[2])));
  if (cw < 1 || ch < 1) return null;
  const rot = ch / cw >= 1.5;
  const W = rot ? ch : cw, H = rot ? cw : ch;
  if (W < 4 || H < 4) return null;

  // 1. straighten the box into a cw x ch picture (cubic interpolation, edges repeated),
  //    stored as B, G, R bytes: the same as cv2.warpPerspective(..., INTER_CUBIC)
  const crop = new Uint8ClampedArray(cw * ch * 3);
  const [p0, p1, p2, p3] = pts;
  const kx = new Float64Array(4), ky = new Float64Array(4), xi = new Int32Array(4), yi = new Int32Array(4);
  for (let y = 0, o = 0; y < ch; y++) {
    const t = y / ch;
    for (let x = 0; x < cw; x++, o += 3) {
      const s = x / cw;
      const sx = Math.round((p0[0] + s * (p1[0] - p0[0]) + t * (p3[0] - p0[0]) + s * t * (p0[0] - p1[0] + p2[0] - p3[0])) * 32) / 32;
      const sy = Math.round((p0[1] + s * (p1[1] - p0[1]) + t * (p3[1] - p0[1]) + s * t * (p0[1] - p1[1] + p2[1] - p3[1])) * 32) / 32;
      const ix = Math.floor(sx), iy = Math.floor(sy);
      cubic(sx - ix, kx);
      cubic(sy - iy, ky);
      for (let i = 0; i < 4; i++) {
        xi[i] = clamp(ix - 1 + i, 0, w - 1) * 4;
        yi[i] = clamp(iy - 1 + i, 0, h - 1) * w * 4;
      }
      let b = 0, g = 0, r = 0;
      for (let j = 0; j < 4; j++) {
        const row = yi[j], wy = ky[j];
        for (let i = 0; i < 4; i++) {
          const p = row + xi[i], wt = wy * kx[i];
          r += rgba[p] * wt;
          g += rgba[p + 1] * wt;
          b += rgba[p + 2] * wt;
        }
      }
      crop[o] = b;
      crop[o + 1] = g;
      crop[o + 2] = r;
    }
  }

  // 2. scale it to 48 px high (bilinear, like cv2.resize), turning sideways text upright
  const ratio = W / H;
  const width = Math.trunc(48 * Math.max(320 / 48, ratio));   // at least 320, padded with grey
  const rw = Math.min(width, Math.ceil(48 * ratio));
  const plane = 48 * width, data = new Float32Array(3 * plane);
  const at = (rx, ry) => (rot ? (rx * cw + (cw - 1 - ry)) * 3 : (ry * cw + rx) * 3);   // upright (rx, ry) -> crop
  for (let oy = 0; oy < 48; oy++) {
    const ry = clamp(((oy + 0.5) * H) / 48 - 0.5, 0, H - 1), ya = Math.floor(ry), yb = Math.min(ya + 1, H - 1), fy = ry - ya;
    for (let ox = 0; ox < rw; ox++) {
      const rx = clamp(((ox + 0.5) * W) / rw - 0.5, 0, W - 1), xa = Math.floor(rx), xb = Math.min(xa + 1, W - 1), fx = rx - xa;
      const p00 = at(xa, ya), p01 = at(xb, ya), p10 = at(xa, yb), p11 = at(xb, yb);
      const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
      const o = oy * width + ox;
      for (let c = 0; c < 3; c++) {
        // cv2.resize gives whole bytes, so round before scaling to -1..1
        const v = Math.round(crop[p00 + c] * w00 + crop[p01 + c] * w01 + crop[p10 + c] * w10 + crop[p11 + c] * w11);
        data[c * plane + o] = v / 127.5 - 1;
      }
    }
  }
  return { data, width };
}

/** Cubic interpolation weights for the 4 pixels around a point (OpenCV's, A = -0.75). */
function cubic(x, k) {
  const A = -0.75;
  k[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
  k[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
  k[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
  k[3] = 1 - k[0] - k[1] - k[2];
}

/**
 * The recogniser's output for one line (T time steps x C characters, each row sums to 1)
 * -> { text, conf, frames }. frames keeps the top 4 characters per step as [char, prob]
 * ("" is the blank), which the Python rules use to re-read a field under a pattern.
 */
export function decodeLine(probs, T, C, chars, k = 4, keep = 0.005) {
  let text = "", sum = 0, n = 0, prev = 0;
  const frames = [], idx = new Int32Array(k), val = new Float32Array(k);
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let m = 0;
    for (let c = 0; c < C; c++) {            // keep the k largest, in order
      const p = probs[off + c];
      if (m < k || p > val[m - 1]) {
        let j = m < k ? m++ : m - 1;
        while (j > 0 && val[j - 1] < p) {
          val[j] = val[j - 1];
          idx[j] = idx[j - 1];
          j--;
        }
        val[j] = p;
        idx[j] = c;
      }
    }
    if (idx[0] !== 0 && idx[0] !== prev) {   // CTC: skip blanks and repeats
      text += chars[idx[0]];
      sum += val[0];
      n++;
    }
    prev = idx[0];
    const row = [];
    for (let i = 0; i < m; i++) {
      if (val[i] < keep && row.length) break;
      row.push([chars[idx[i]], Math.round(val[i] * 1e4) / 1e4]);
      if (val[i] > 0.995) break;
    }
    frames.push(row);
  }
  return { text, conf: n ? sum / n : 0, frames };
}
