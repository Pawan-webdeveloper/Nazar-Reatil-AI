/* Minimal dependency-free SVG charts (works offline on the edge box).
   barChart, lineChart (1-2 series, crosshair tooltip), matrixChart (sequential heat grid). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tooltip");
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function showTip(evt, html) {
    const t = tip(); t.innerHTML = html; t.classList.add("show");
    const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
    const y = Math.max(8, evt.clientY - t.offsetHeight - 10);
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function hideTip() { tip().classList.remove("show"); }
  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }
  const fmt = (v) => (v == null ? "-" : Math.abs(v) >= 1000 ? v.toLocaleString() : (Math.round(v * 10) / 10).toString());
  function frame(container, pad) {
    container.innerHTML = "";
    const w = container.clientWidth || 600, h = container.clientHeight || 240;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, container);
    return { svg, w, h, iw: w - pad.l - pad.r, ih: h - pad.t - pad.b };
  }
  function empty(container, msg) {
    container.innerHTML = `<div class="empty-state">
      <svg viewBox="0 0 64 48" aria-hidden="true"><path d="M8 40h48"/><rect x="14" y="26" width="7" height="14" rx="2"/>
      <rect x="28" y="18" width="7" height="22" rx="2"/><rect x="42" y="30" width="7" height="10" rx="2"/>
      <path d="M50 10l3-4M54 14l5-1M46 8l-1-5"/></svg>
      <strong>No data yet</strong><span>${msg || "Data will appear once the cameras start sending results for this period."}</span></div>`;
  }
  window.emptyState = empty;

  // tiny trend line for KPI cards (no axes; tooltip via title)
  window.sparkline = function (values, color) {
    const v = values.map((x) => (x == null ? null : Number(x)));
    const pts = v.filter((x) => x != null);
    if (pts.length < 2) return "";
    const w = 72, h = 30, max = Math.max(...pts), min = Math.min(...pts), rng = max - min || 1;
    let d = "", pen = false;
    v.forEach((x, i) => {
      if (x == null) { pen = false; return; }
      const X = (i / (v.length - 1)) * (w - 4) + 2, Y = h - 3 - ((x - min) / rng) * (h - 6);
      d += (pen ? "L" : "M") + X.toFixed(1) + "," + Y.toFixed(1) + " "; pen = true;
    });
    return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  };
  function yAxis(g, f, pad, max, ticks) {
    for (let i = 0; i <= ticks; i++) {
      const v = (max / ticks) * i, y = pad.t + f.ih - (v / max) * f.ih;
      el("line", { x1: pad.l, x2: pad.l + f.iw, y1: y, y2: y, class: i === 0 ? "axis" : "gridline" }, g);
      const t = el("text", { x: pad.l - 6, y: y + 4, "text-anchor": "end" }, g);
      t.textContent = fmt(v);
    }
  }

  window.barChart = function (container, labels, values, opts = {}) {
    if (!values.length || values.every((v) => !v)) return empty(container, opts.empty);
    const pad = { l: 44, r: 8, t: 10, b: 26 };
    const f = frame(container, pad);
    const max = niceMax(Math.max(...values));
    yAxis(f.svg, f, pad, max, 4);
    const bw = f.iw / values.length, gap = Math.max(2, bw * 0.18);
    const every = Math.ceil(values.length / Math.max(1, Math.floor(f.iw / 42)));
    values.forEach((v, i) => {
      const x = pad.l + i * bw + gap / 2, bh = (v / max) * f.ih, y = pad.t + f.ih - bh;
      const w = Math.max(1, bw - gap), r = Math.min(4, w / 2, bh);
      // rounded data end, flat at the baseline
      const d = `M${x},${pad.t + f.ih} V${y + r} Q${x},${y} ${x + r},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${pad.t + f.ih} Z`;
      el("path", { d, fill: opts.color || "var(--series-1)" }, f.svg);
      const hit = el("rect", { x: pad.l + i * bw, y: pad.t, width: bw, height: f.ih, fill: "transparent" }, f.svg);
      hit.addEventListener("mousemove", (e) => showTip(e, `<b>${labels[i]}</b><br>${fmt(v)} ${opts.unit || ""}`));
      hit.addEventListener("mouseleave", hideTip);
      if (i % every === 0) {
        const t = el("text", { x: x + w / 2, y: f.h - 8, "text-anchor": "middle" }, f.svg);
        t.textContent = labels[i];
      }
    });
  };

  window.lineChart = function (container, labels, series, opts = {}) {
    // series: [{name, values, color}]
    const all = series.flatMap((s) => s.values.filter((v) => v != null));
    if (!all.length) return empty(container, opts.empty);
    const pad = { l: 44, r: 12, t: 10, b: 26 };
    const f = frame(container, pad);
    const max = niceMax(Math.max(...all, opts.minMax || 0));
    yAxis(f.svg, f, pad, max, 4);
    const n = labels.length;
    const X = (i) => pad.l + (n <= 1 ? f.iw / 2 : (i / (n - 1)) * f.iw);
    const Y = (v) => pad.t + f.ih - (v / max) * f.ih;
    const every = Math.ceil(n / Math.max(1, Math.floor(f.iw / 70)));
    labels.forEach((l, i) => {
      if (i % every === 0) { const t = el("text", { x: X(i), y: f.h - 8, "text-anchor": "middle" }, f.svg); t.textContent = l; }
    });
    series.forEach((s) => {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + X(i) + "," + Y(v) + " "; pen = true;
      });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dashed ? "5 4" : "none" }, f.svg);
      s.values.forEach((v, i) => { if (v != null && s.markers) el("circle", { cx: X(i), cy: Y(v), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }, f.svg); });
    });
    const cross = el("line", { y1: pad.t, y2: pad.t + f.ih, class: "axis", opacity: 0 }, f.svg);
    const overlay = el("rect", { x: pad.l, y: pad.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
    overlay.addEventListener("mousemove", (e) => {
      const r = f.svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * f.w;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - pad.l) / f.iw) * (n - 1))));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 1);
      const rows = series.filter((s) => s.values[i] != null)
        .map((s) => `<span style="display:inline-block;width:10px;height:3px;background:${s.color};margin-right:5px;vertical-align:middle"></span>${s.name}: <b>${fmt(s.values[i])}</b>`);
      if (rows.length) showTip(e, `<b>${labels[i]}</b><br>${rows.join("<br>")}`); else hideTip();
    });
    overlay.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("opacity", 0); });
  };

  window.matrixChart = function (container, rowLabels, colLabels, matrix, opts = {}) {
    const flat = matrix.flat();
    if (!flat.length || flat.every((v) => !v)) return empty(container, opts.empty);
    const pad = { l: 40, r: 8, t: 6, b: 22 };
    const f = frame(container, pad);
    const max = Math.max(...flat);
    const cw = f.iw / colLabels.length, ch = f.ih / rowLabels.length;
    const steps = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5", "--seq-6", "--seq-7"];
    matrix.forEach((row, r) => {
      const t = el("text", { x: pad.l - 6, y: pad.t + r * ch + ch / 2 + 4, "text-anchor": "end" }, f.svg); t.textContent = rowLabels[r];
      row.forEach((v, c) => {
        const k = v <= 0 ? 0 : Math.min(7, 1 + Math.floor((v / max) * 6.999));
        const cell = el("rect", { x: pad.l + c * cw + 1, y: pad.t + r * ch + 1, width: Math.max(1, cw - 2), height: Math.max(1, ch - 2),
          rx: 2, fill: `var(${steps[k]})` }, f.svg);
        cell.addEventListener("mousemove", (e) => showTip(e, `<b>${rowLabels[r]} ${colLabels[c]}</b><br>${fmt(v)} ${opts.unit || ""}`));
        cell.addEventListener("mouseleave", hideTip);
      });
    });
    colLabels.forEach((l, c) => {
      if (c % 3 === 0) { const t = el("text", { x: pad.l + c * cw + cw / 2, y: f.h - 6, "text-anchor": "middle" }, f.svg); t.textContent = l; }
    });
  };
})();
