// ==UserScript==
// @name         Ghost Font Live Canvas Analyzer
// @namespace    https://local.example/
// @version      1.0.0
// @description  Mixfont Ghost Font canvasını yalnızca piksel hareketlerinden canlı analiz eder.
// @match        https://www.mixfont.com/ghost-font*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(() => {
  "use strict";

  const SOURCE_SELECTOR = 'canvas[aria-label="Ghost Font animation preview"]';
  const ANALYSIS_W = 640;
  const ANALYSIS_H = 360;
  const SAMPLE_MS = 45;
  const HISTORY = 32;
  const MAX_DY = 18;
  const DECAY = 0.92;

  let running = true;
  let mode = "combined";
  let invert = false;
  let cropTop = 0.20;
  let cropBottom = 0.80;

  let sourceCanvas = null;
  let sampleCanvas = null;
  let sampleCtx = null;
  let outputCanvas = null;
  let outputCtx = null;
  let statusEl = null;

  let previous = null;
  let previousAligned = null;
  let cumulativeY = 0;
  let energy = null;
  let history = [];
  let timer = null;
  let frameNo = 0;

  function waitForCanvas() {
    sourceCanvas = document.querySelector(SOURCE_SELECTOR);
    if (!sourceCanvas) {
      setTimeout(waitForCanvas, 500);
      return;
    }
    createUI();
    resetAnalysis();
    timer = setInterval(processFrame, SAMPLE_MS);
  }

  function createUI() {
    const panel = document.createElement("section");
    panel.id = "ghost-live-analyzer";
    panel.innerHTML = `
      <style>
        #ghost-live-analyzer {
          position: fixed;
          z-index: 2147483647;
          right: 14px;
          bottom: 14px;
          width: min(680px, calc(100vw - 28px));
          padding: 12px;
          border-radius: 14px;
          background: rgba(10, 12, 18, .94);
          color: #fff;
          box-shadow: 0 14px 50px rgba(0,0,0,.45);
          font-family: Arial, sans-serif;
          backdrop-filter: blur(12px);
        }
        #ghost-live-analyzer .gf-head {
          display:flex; align-items:center; justify-content:space-between;
          gap:10px; margin-bottom:9px;
        }
        #ghost-live-analyzer .gf-title {font-weight:700;font-size:15px}
        #ghost-live-analyzer .gf-status {font-size:12px;opacity:.75}
        #ghost-live-analyzer canvas {
          display:block; width:100%; aspect-ratio:16/9;
          background:#111; border-radius:9px; image-rendering:auto;
        }
        #ghost-live-analyzer .gf-tools {
          display:flex; flex-wrap:wrap; gap:7px; margin-top:9px;
        }
        #ghost-live-analyzer button, #ghost-live-analyzer select {
          border:1px solid rgba(255,255,255,.18);
          background:#232631; color:#fff; border-radius:8px;
          padding:7px 10px; cursor:pointer;
        }
        #ghost-live-analyzer label {
          display:flex; align-items:center; gap:6px;
          font-size:12px; background:#191c24; padding:5px 8px;
          border-radius:8px;
        }
        #ghost-live-analyzer input[type=range] { width:90px; }
        #ghost-live-analyzer .gf-close {font-size:18px;padding:2px 8px}
      </style>
      <div class="gf-head">
        <div>
          <div class="gf-title">Ghost Font — Canlı Canvas Analizi</div>
          <div class="gf-status">Hazırlanıyor…</div>
        </div>
        <button class="gf-close" title="Paneli kapat">×</button>
      </div>
      <canvas width="${ANALYSIS_W}" height="${ANALYSIS_H}"></canvas>
      <div class="gf-tools">
        <button data-action="toggle">Duraklat</button>
        <button data-action="reset">Sıfırla</button>
        <button data-action="invert">Ters Çevir</button>
        <select data-action="mode">
          <option value="combined">Birleşik</option>
          <option value="energy">Hareket Enerjisi</option>
          <option value="range">Zamansal Aralık</option>
          <option value="difference">Kare Farkı</option>
        </select>
        <label>Üst
          <input data-action="top" type="range" min="0" max="45" value="20">
        </label>
        <label>Alt
          <input data-action="bottom" type="range" min="55" max="100" value="80">
        </label>
      </div>
    `;
    document.body.appendChild(panel);

    outputCanvas = panel.querySelector("canvas");
    outputCtx = outputCanvas.getContext("2d", { alpha: false });
    statusEl = panel.querySelector(".gf-status");

    sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = ANALYSIS_W;
    sampleCanvas.height = ANALYSIS_H;
    sampleCtx = sampleCanvas.getContext("2d", {
      alpha: false,
      willReadFrequently: true
    });

    panel.querySelector('[data-action="toggle"]').onclick = (e) => {
      running = !running;
      e.currentTarget.textContent = running ? "Duraklat" : "Devam Et";
    };
    panel.querySelector('[data-action="reset"]').onclick = resetAnalysis;
    panel.querySelector('[data-action="invert"]').onclick = () => {
      invert = !invert;
    };
    panel.querySelector('[data-action="mode"]').onchange = (e) => {
      mode = e.target.value;
    };
    panel.querySelector('[data-action="top"]').oninput = (e) => {
      cropTop = Number(e.target.value) / 100;
      if (cropTop >= cropBottom - .05) cropTop = cropBottom - .05;
      resetAnalysis();
    };
    panel.querySelector('[data-action="bottom"]').oninput = (e) => {
      cropBottom = Number(e.target.value) / 100;
      if (cropBottom <= cropTop + .05) cropBottom = cropTop + .05;
      resetAnalysis();
    };
    panel.querySelector(".gf-close").onclick = () => {
      clearInterval(timer);
      panel.remove();
    };
  }

  function resetAnalysis() {
    previous = null;
    previousAligned = null;
    cumulativeY = 0;
    energy = new Float32Array(ANALYSIS_W * ANALYSIS_H);
    history = [];
    frameNo = 0;
    if (outputCtx) {
      outputCtx.fillStyle = "#111";
      outputCtx.fillRect(0, 0, ANALYSIS_W, ANALYSIS_H);
    }
  }

  function getGrayFrame() {
    sampleCtx.drawImage(sourceCanvas, 0, 0, ANALYSIS_W, ANALYSIS_H);
    const rgba = sampleCtx.getImageData(0, 0, ANALYSIS_W, ANALYSIS_H).data;
    const gray = new Uint8Array(ANALYSIS_W * ANALYSIS_H);
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
      gray[p] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
    }
    return gray;
  }

  function estimateVerticalShift(a, b) {
    const y0 = Math.max(2, Math.floor(ANALYSIS_H * cropTop));
    const y1 = Math.min(ANALYSIS_H - 2, Math.ceil(ANALYSIS_H * cropBottom));
    const x0 = Math.floor(ANALYSIS_W * 0.08);
    const x1 = Math.ceil(ANALYSIS_W * 0.92);
    const xStep = 4;
    const yStep = 3;

    let bestDy = 0;
    let bestScore = Number.POSITIVE_INFINITY;

    for (let dy = -MAX_DY; dy <= MAX_DY; dy++) {
      let score = 0;
      let count = 0;

      for (let y = y0; y < y1; y += yStep) {
        const by = y + dy;
        if (by < y0 || by >= y1) continue;

        const rowA = y * ANALYSIS_W;
        const rowB = by * ANALYSIS_W;
        for (let x = x0; x < x1; x += xStep) {
          // Yatay doku baskısını azaltmak için dikey türev karşılaştırması.
          const ga = a[rowA + x] - a[rowA - ANALYSIS_W + x];
          const gb = b[rowB + x] - b[rowB - ANALYSIS_W + x];
          score += Math.abs(ga - gb);
          count++;
        }
      }

      if (count && score / count < bestScore) {
        bestScore = score / count;
        bestDy = dy;
      }
    }
    return bestDy;
  }

  function shiftVertical(src, shiftY) {
    const out = new Uint8Array(src.length);
    const rounded = Math.round(shiftY);
    for (let y = 0; y < ANALYSIS_H; y++) {
      const sy = y + rounded;
      if (sy < 0 || sy >= ANALYSIS_H) continue;
      out.set(
        src.subarray(sy * ANALYSIS_W, (sy + 1) * ANALYSIS_W),
        y * ANALYSIS_W
      );
    }
    return out;
  }

  function temporalRange() {
    const out = new Float32Array(ANALYSIS_W * ANALYSIS_H);
    if (history.length < 2) return out;

    for (let p = 0; p < out.length; p++) {
      let lo = 255;
      let hi = 0;
      for (let k = 0; k < history.length; k++) {
        const v = history[k][p];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      out[p] = hi - lo;
    }
    return out;
  }

  function percentile(values, q) {
    const sample = [];
    const step = Math.max(1, Math.floor(values.length / 18000));
    for (let i = 0; i < values.length; i += step) sample.push(values[i]);
    sample.sort((a, b) => a - b);
    return sample[Math.min(sample.length - 1, Math.floor(q * sample.length))] || 1;
  }

  function render(values) {
    const img = outputCtx.createImageData(ANALYSIS_W, ANALYSIS_H);
    const low = percentile(values, 0.08);
    const high = Math.max(low + 1, percentile(values, 0.985));
    const y0 = Math.floor(ANALYSIS_H * cropTop);
    const y1 = Math.ceil(ANALYSIS_H * cropBottom);

    for (let p = 0; p < values.length; p++) {
      const y = Math.floor(p / ANALYSIS_W);
      let v = 0;

      if (y >= y0 && y <= y1) {
        v = (values[p] - low) * 255 / (high - low);
        v = Math.max(0, Math.min(255, v));
        // Harf izlerini belirginleştiren yumuşak gamma.
        v = 255 * Math.pow(v / 255, 0.72);
      }

      if (invert) v = 255 - v;
      const j = p * 4;
      img.data[j] = v;
      img.data[j + 1] = v;
      img.data[j + 2] = v;
      img.data[j + 3] = 255;
    }
    outputCtx.putImageData(img, 0, 0);
  }

  function processFrame() {
    if (!running || !sourceCanvas || !document.contains(sourceCanvas)) return;

    try {
      const current = getGrayFrame();

      if (!previous) {
        previous = current;
        previousAligned = current;
        history.push(current);
        statusEl.textContent = "İlk kare alındı";
        return;
      }

      const dy = estimateVerticalShift(previous, current);
      cumulativeY += dy;
      cumulativeY = Math.max(-45, Math.min(45, cumulativeY));

      const aligned = shiftVertical(current, cumulativeY);
      const difference = new Float32Array(aligned.length);

      for (let p = 0; p < aligned.length; p++) {
        const d = Math.abs(aligned[p] - previousAligned[p]);
        difference[p] = d;
        energy[p] = energy[p] * DECAY + d * (1 - DECAY);
      }

      history.push(aligned);
      if (history.length > HISTORY) history.shift();

      let display;
      if (mode === "energy") {
        display = energy;
      } else if (mode === "range") {
        display = temporalRange();
      } else if (mode === "difference") {
        display = difference;
      } else {
        const range = temporalRange();
        display = new Float32Array(aligned.length);
        for (let p = 0; p < display.length; p++) {
          display[p] = energy[p] * 0.62 + range[p] * 0.38;
        }
      }

      render(display);
      previous = current;
      previousAligned = aligned;
      frameNo++;
      statusEl.textContent =
        `Kare: ${frameNo} · anlık ΔY: ${dy}px · toplam hizalama: ${cumulativeY}px`;
    } catch (err) {
      statusEl.textContent = `Hata: ${err.message}`;
      console.error("Ghost analyzer:", err);
    }
  }

  waitForCanvas();
})();