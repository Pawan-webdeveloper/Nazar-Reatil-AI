// Builds the SIH-style deck for RetailSense AI.
// node build.js  ->  <project>/presentation/RetailSense_AI_SIH_PPT.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa6");
const si = require("react-icons/si");

const PROJECT = "C:/Users/itanm/OneDrive/Desktop/camera inventory";
const A = (f) => path.join(PROJECT, "presentation", "assets", f);
const OUT = process.env.OUT || path.join(PROJECT, "presentation", "RetailSense_AI_SIH_PPT.pptx");

// ---------------------------------------------------------------- palette & fonts
const C = {
  bg: "F2F4F7", card: "FFFFFF", line: "CBD3DC", ink: "111111", muted: "4A5560",
  red: "C0171D", green: "1E6B3A", brand: "134E4A", orange: "E8761B", blue: "1F3A93",
  chip: "C6EFCE", chipLine: "2E7D32", banner: "D32F2F", purple: "6B4FA0",
};
const HEAD = "Cambria", BODY = "Calibri";

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const png = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

// "Label | text with **highlight**"  ->  pptxgenjs runs for one paragraph
function runs(str, opts = {}) {
  const out = [];
  let label = null, rest = str;
  const bar = str.indexOf(" | ");
  if (bar > -1) { label = str.slice(0, bar); rest = str.slice(bar + 3); }
  if (label) out.push({ text: label + ": ", options: { bold: true, color: C.ink } });
  rest.split("**").forEach((part, i) => {
    if (!part) return;
    out.push({ text: part, options: i % 2 ? { bold: true, color: opts.hl || C.red } : { color: opts.color || C.ink } });
  });
  return out;
}
function paragraphs(items, { bullet = true, size = 12, gap = 4, hl, color, numbered = false } = {}) {
  const all = [];
  items.forEach((it, idx) => {
    const r = runs(it, { hl, color });
    r[0].options = { ...r[0].options, bullet: numbered ? { type: "number" } : bullet ? true : false, paraSpaceAfter: gap };
    if (idx < items.length - 1) r[r.length - 1].options = { ...r[r.length - 1].options, breakLine: true };
    all.push(...r);
  });
  return all.map((x) => ({ text: x.text, options: { fontFace: BODY, fontSize: size, ...x.options } }));
}

function header(pres, slide, title) {
  slide.background = { color: C.bg };
  slide.addShape(pres.shapes.OVAL, { x: 0.25, y: 0.12, w: 1.35, h: 0.62, fill: { color: C.card }, line: { color: C.purple, width: 1.25 } });
  slide.addText("[Team Name]", { x: 0.25, y: 0.12, w: 1.35, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: 10, color: C.ink, isTextBox: true, margin: 0 });
  if (title) slide.addText(title, { x: 1.8, y: 0.1, w: 9.7, h: 0.7, align: "center", valign: "middle", fontFace: HEAD, fontSize: 30, bold: true, color: C.ink, isTextBox: true });
  slide.addText([{ text: "SMART INDIA", options: { breakLine: true } }, { text: "HACKATHON 2025" }],
    { x: 11.35, y: 0.14, w: 1.75, h: 0.6, align: "right", valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: "333333", isTextBox: true, margin: 0 });
}

function card(pres, slide, x, y, w, h, heading, ic, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.card }, line: { color: C.line, width: 0.75 } });
  if (heading) {
    let tx = x + 0.15;
    if (ic) {
      slide.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.1, w: 0.36, h: 0.36, fill: { color: opts.icBg || "E8F1EE" }, line: { color: opts.icBg || "E8F1EE", width: 0 } });
      slide.addImage({ data: ic, x: x + 0.22, y: y + 0.17, w: 0.22, h: 0.22 });
      tx = x + 0.6;
    }
    slide.addText(heading, { x: tx, y: y + 0.07, w: w - (tx - x) - 0.15, h: 0.42, fontFace: HEAD, fontSize: opts.hSize || 16, bold: true, color: opts.hColor || C.ink, valign: "middle", isTextBox: true, margin: 0 });
  }
}

function arrow(pres, slide, x, y, w, h, color = "555555") {
  slide.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width: 1.25, endArrowType: "triangle" } });
}
function box(pres, slide, x, y, w, h, text, fill, line, size = 9, shape) {
  const sh = shape || pres.shapes.ROUNDED_RECTANGLE;
  const o = { x, y, w, h, fill: { color: fill }, line: { color: line, width: 1 } };
  if (sh === pres.shapes.ROUNDED_RECTANGLE) o.rectRadius = 0.05;  // rectRadius on other shapes corrupts the file
  slide.addShape(sh, o);
  slide.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: size, bold: true, color: C.ink, isTextBox: true, margin: 2 });
}

const WANT = process.env.SLIDES ? process.env.SLIDES.split(",").map(Number) : null;
const want = (n) => !WANT || WANT.includes(n);
(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.title = "RetailSense AI - Smart India Hackathon 2025";

  const I = {
    warn: await icon(fa.FaTriangleExclamation, "C0171D"), bulb: await icon(fa.FaLightbulb, "E8761B"),
    star: await icon(fa.FaStar, "1E6B3A"), gears: await icon(fa.FaGears, "134E4A"), layers: await icon(fa.FaLayerGroup, "134E4A"),
    flow: await icon(fa.FaDiagramProject, "134E4A"), chart: await icon(fa.FaChartLine, "1F3A93"), store: await icon(fa.FaStore, "134E4A"),
    users: await icon(fa.FaUsers, "1F3A93"), coins: await icon(fa.FaCoins, "B7791F"), shield: await icon(fa.FaShieldHalved, "134E4A"),
    book: await icon(fa.FaBookOpen, "C0171D"), flask: await icon(fa.FaFlask, "1F3A93"), check: await icon(fa.FaCircleCheck, "1E6B3A"),
    video: await icon(fa.FaVideo, "134E4A"),
  };
  const S = {
    python: await icon(si.SiPython, "3776AB"), torch: await icon(si.SiPytorch, "EE4C2C"), yolo: await icon(si.SiUltralytics, "111F68"),
    opencv: await icon(si.SiOpencv, "5C3EE8"), onnx: await icon(si.SiOnnx, "005CED"), sklearn: await icon(si.SiScikitlearn, "F7931E"),
    fastapi: await icon(si.SiFastapi, "009688"), sqlite: await icon(si.SiSqlite, "003B57"), docker: await icon(si.SiDocker, "2496ED"),
    js: await icon(si.SiJavascript, "C9A400"), colab: await icon(si.SiGooglecolab, "F9AB00"), pandas: await icon(si.SiPandas, "150458"),
  };

  // =================================================================== 1. TITLE
  if (want(1)) {
    const s = pres.addSlide();
    s.background = { color: C.bg };
    s.addText("SMART INDIA HACKATHON 2025", { x: 0.5, y: 0.35, w: 12.33, h: 0.5, align: "center", fontFace: HEAD, fontSize: 22, bold: true, color: C.orange, isTextBox: true });
    s.addText([
      { text: "नज़र", options: { fontFace: "Nirmala UI", color: "138808" } },
      { text: " – ", options: { color: C.ink } },
      { text: "RetailSense AI", options: { fontFace: HEAD, color: C.orange } },
    ], { x: 0.5, y: 0.95, w: 12.33, h: 1.0, align: "center", fontSize: 48, bold: true, isTextBox: true });
    s.addText("Edge-AI Retail Intelligence Platform: Shopper Analytics, Shelf Monitoring & Queue Prediction on In-Store Cameras",
      { x: 1.0, y: 1.95, w: 11.33, h: 0.6, align: "center", fontFace: BODY, fontSize: 18, bold: true, color: C.blue, isTextBox: true });

    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.95, w: 6.1, h: 3.9, rectRadius: 0.1, fill: { color: C.card }, line: { color: C.line, width: 0.75 } });
    const rows = [["Problem Statement ID", "[Enter PS ID]"], ["Problem Statement Title", "AI-powered retail intelligence with on-device (edge) AI"],
      ["Theme", "[Enter Theme]"], ["PS Category", "Software"], ["Team ID", "[Enter Team ID]"], ["Team Name", "[Enter Team Name]"]];
    rows.forEach(([k, v], i) => {
      const y = 3.15 + i * 0.6;
      s.addText(k, { x: 0.85, y, w: 2.4, h: 0.5, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, valign: "middle", isTextBox: true, margin: 0 });
      s.addText(v, { x: 3.3, y, w: 3.25, h: 0.5, fontFace: BODY, fontSize: 14, color: v.startsWith("[") ? C.red : C.muted, valign: "middle", isTextBox: true, margin: 0 });
    });
    s.addImage({ path: A("ui_overview.png"), x: 7.0, y: 2.95, w: 5.9, h: 3.69, shadow: { type: "outer", blur: 8, offset: 3, angle: 90, color: "000000", opacity: 0.25 } });
    s.addText("Working prototype: live store dashboard", { x: 7.0, y: 6.7, w: 5.9, h: 0.35, align: "center", fontFace: BODY, fontSize: 12, italic: true, color: C.muted, isTextBox: true });
    s.addNotes("Title slide. Apni team ki details (PS ID, Theme, Team ID, Team Name) laal brackets wali jagah par bharna. Ek line mein bolna: RetailSense AI store ke CCTV ko smart banata hai - bheed, khaali shelf aur billing queue ko store ke andar hi AI se samajhta hai, bina internet ke.");
  }

  // =================================================================== 2. IDEA
  if (want(2)) {
    const s = pres.addSlide();
    header(pres, s, null);
    s.addText([
      { text: "नज़र", options: { fontFace: "Nirmala UI", color: "138808" } },
      { text: " – ", options: { color: C.ink } },
      { text: "RetailSense AI", options: { fontFace: HEAD, color: C.orange } },
    ], { x: 1.8, y: 0.05, w: 9.7, h: 0.62, align: "center", fontSize: 30, bold: true, isTextBox: true });
    s.addText("Edge-AI Retail Intelligence for Indian Stores", { x: 1.8, y: 0.62, w: 9.7, h: 0.34, align: "center", fontFace: BODY, fontSize: 15, bold: true, color: C.blue, isTextBox: true });

    card(pres, s, 0.25, 1.1, 6.45, 2.62, "Understanding the Problem", I.warn, { icBg: "FBE3E4" });
    s.addText(paragraphs([
      "Empty shelves go unnoticed | about **8% of items are out of stock** at any time and shoppers **buy elsewhere**.",
      "Long billing queues | managers see the rush **too late**; shoppers **leave the line**.",
      "No shopper insights | stores **don't know footfall, peak hours** or which displays work.",
      "Cloud CCTV analytics is costly | needs **fast internet** and **sends video outside** (privacy risk).",
      "Weak internet in Tier-2/3 towns | cloud tools **stop working during outages**.",
    ], { size: 12.5, gap: 5 }), { x: 0.4, y: 1.58, w: 6.15, h: 2.1, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 0.25, 3.82, 6.45, 3.1, "Our Solution", I.bulb, { icBg: "FDEBD9" });
    s.addText(paragraphs([
      "Footfall Counting | counts people **entering and leaving** with a virtual line.",
      "Heatmaps & Dwell Time | shows **where shoppers walk and how long they stay**.",
      "Shelf Monitoring | finds **empty slots and low stock** from shelf cameras.",
      "Queue Intelligence | measures **waiting time** and **predicts queues 15 min ahead**.",
      "Smart Alerts | tells staff to **refill a shelf** or **open a counter**.",
      "Edge AI | all AI runs **inside the store**, works **without internet**.",
      "Dashboard & Reports | daily/weekly **footfall, conversion, availability**.",
    ], { size: 12.5, gap: 4 }), { x: 0.4, y: 4.3, w: 6.15, h: 2.58, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 6.85, 1.1, 6.23, 3.0, "Our Unique Solutions", I.star, { icBg: "DDF2E3" });
    s.addText(paragraphs([
      "Works on existing CCTV | uses the **store's own RTSP cameras**.",
      "Runs on a ₹35k mini PC | **no GPU**, 3 cameras on one CPU (measured).",
      "Privacy by design | **no faces or photos stored**, only anonymous counts.",
      "Predicts, not just reports | queue forecast from **entrance footfall**.",
      "\"Mark as Full\" calibration | compares a shelf with **its own full photo**.",
      "Offline-first | stores data locally, **syncs to HQ when online**.",
      "POS / ERP ready | bills via CSV/API, **auto refill orders**.",
    ], { size: 13.5, gap: 4 }), { x: 7.0, y: 1.58, w: 5.95, h: 2.48, valign: "top", isTextBox: true, margin: 0 });

    s.addText("USPs", { x: 6.85, y: 4.28, w: 0.62, h: 0.9, fontFace: HEAD, fontSize: 13, bold: true, color: C.ink, valign: "middle", isTextBox: true, margin: 0 });
    ["Real-time footfall, dwell & heatmap", "Empty-shelf alerts in minutes", "15-min queue forecast + counter advice", "100% on-device, privacy-first"].forEach((t, i) => {
      box(pres, s, 7.47 + i * 1.41, 4.25, 1.33, 0.95, t, C.chip, C.chipLine, 10);
    });

    card(pres, s, 6.85, 5.32, 6.23, 1.6, null);
    s.addImage({ path: A("ui_live.png"), x: 6.97, y: 5.42, w: 2.24, h: 1.4 });
    s.addText("Prototype (working today)", { x: 9.35, y: 5.38, w: 3.65, h: 0.34, fontFace: HEAD, fontSize: 13, bold: true, color: C.green, isTextBox: true, margin: 0 });
    s.addText(paragraphs([
      "**5 live cameras** processed on a laptop CPU",
      "Shelf model **mAP50 0.93** on unseen images",
      "Dashboard, alerts, reports, **33 tests pass**",
    ], { size: 10.5, gap: 2 }), { x: 9.35, y: 5.74, w: 3.65, h: 1.12, valign: "top", isTextBox: true, margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 7.02, w: 13.333, h: 0.48, fill: { color: C.banner }, line: { color: C.banner, width: 0 } });
    s.addText("WORKING PROTOTYPE IS READY – TESTED ON REAL PUBLIC VIDEOS & BENCHMARK DATASETS", { x: 0, y: 7.02, w: 13.333, h: 0.48, align: "center", valign: "middle", fontFace: BODY, fontSize: 18, bold: true, color: "FFFFFF", isTextBox: true });
    s.addNotes("Problem: store mein khaali shelf, lambi billing line aur customer ka data nahi hota. Cloud CCTV mehnga hai aur internet chahiye. Solution: store ke purane CCTV + ek chhota PC. AI andar hi chalta hai: log ginta hai, heatmap banata hai, khaali shelf pakadta hai, queue ka 15 minute pehle anumaan lagata hai, aur staff ko alert bhejta hai. Chehre ya photo save nahi hote.");
  }

  // =================================================================== 3. TECHNICAL APPROACH
  if (want(3)) {
    const s = pres.addSlide();
    header(pres, s, "Technical Approach");

    card(pres, s, 0.25, 0.95, 4.75, 4.15, "Methodology", I.gears);
    s.addText(paragraphs([
      "Capture | read CCTV streams (**RTSP**) at **3–10 FPS** on the in-store edge box.",
      "Detect | **YOLO11** finds **people** and **shelf products** (trained on **SKU-110K**, 11,743 shelf photos).",
      "Track | **ByteTrack** gives each shopper an **anonymous ID** (no face data).",
      "Analyse | line counting, **zone dwell**, **heatmap**, queue length, **empty-slot** detection.",
      "Predict | **LightGBM** forecasts the queue **15 min ahead**; **Erlang-C** suggests counters.",
      "Act | alerts + dashboard; data kept in **SQLite (offline)** and synced to HQ.",
    ], { size: 12.5, gap: 6, numbered: true }), { x: 0.4, y: 1.45, w: 4.5, h: 3.6, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 0.25, 5.2, 4.75, 2.05, "Tech Stack", I.layers);
    const stack = [["Python", S.python], ["PyTorch", S.torch], ["YOLO11", S.yolo], ["OpenCV", S.opencv], ["ONNX", S.onnx], ["LightGBM / sklearn", S.sklearn],
      ["FastAPI", S.fastapi], ["SQLite", S.sqlite], ["Docker", S.docker], ["JS Dashboard", S.js], ["Colab T4 GPU", S.colab], ["Pandas", S.pandas]];
    stack.forEach(([name, img], i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = 0.4 + col * 1.52, y = 5.68 + row * 0.38;
      s.addImage({ data: img, x, y: y + 0.04, w: 0.26, h: 0.26 });
      s.addText(name, { x: x + 0.3, y, w: 1.2, h: 0.34, fontFace: BODY, fontSize: 10, bold: true, color: C.ink, valign: "middle", isTextBox: true, margin: 0 });
    });

    s.addText("User Interface (store dashboard)", { x: 5.15, y: 0.92, w: 4.25, h: 0.34, fontFace: HEAD, fontSize: 14, bold: true, color: C.ink, isTextBox: true, margin: 0 });
    s.addImage({ path: A("ui_overview.png"), x: 5.15, y: 1.28, w: 4.25, h: 2.66, shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.2 } });
    s.addText("Live Cameras (edge AI output, privacy-blurred)", { x: 5.15, y: 4.05, w: 4.25, h: 0.34, fontFace: HEAD, fontSize: 14, bold: true, color: C.ink, isTextBox: true, margin: 0 });
    s.addImage({ path: A("ui_live.png"), x: 5.15, y: 4.41, w: 4.25, h: 2.66, shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.2 } });

    // ---- implementation flow
    card(pres, s, 9.55, 0.95, 3.53, 6.3, "Implementation Flow", I.flow);
    const X0 = 9.7, W = 3.23, cx = X0 + W / 2;
    ["Entrance cam", "Floor cam", "Checkout cam", "Shelf cam"].forEach((t, i) => box(pres, s, X0 + i * 0.815, 1.5, 0.76, 0.5, t, "E3F2FD", "1976D2", 8.5));
    arrow(pres, s, cx, 2.02, 0, 0.23);
    box(pres, s, X0 + 0.35, 2.27, W - 0.7, 0.42, "Frame sampler (3–10 FPS)", "F5F5F5", "757575", 9.5);
    arrow(pres, s, cx, 2.71, 0, 0.23);
    box(pres, s, X0 + 0.15, 2.96, W - 0.3, 0.46, "Edge AI: YOLO11 detect + ByteTrack", "FFF3E0", "E8761B", 9.5);
    arrow(pres, s, X0 + 0.54, 3.44, 0, 0.24); arrow(pres, s, cx, 3.44, 0, 0.24); arrow(pres, s, X0 + W - 0.54, 3.44, 0, 0.24);
    ["Footfall, dwell, heatmap", "Queue + 15-min forecast", "Shelf empty slots"].forEach((t, i) => box(pres, s, X0 + i * 1.09, 3.7, 1.03, 0.6, t, "E8F5E9", "2E7D32", 8.5));
    arrow(pres, s, cx, 4.32, 0, 0.2);
    box(pres, s, cx - 0.8, 4.54, 1.6, 0.72, "Issue found?", "EDE7F6", "6B4FA0", 9.5, pres.shapes.DIAMOND);
    arrow(pres, s, cx + 0.8, 4.9, 0.3, 0);
    box(pres, s, X0 + W - 0.66, 4.62, 0.66, 0.58, "Alert staff", "FDECEA", "C0171D", 8.5);
    s.addText("Yes", { x: cx + 0.78, y: 4.62, w: 0.4, h: 0.24, fontFace: BODY, fontSize: 8, color: C.red, bold: true, isTextBox: true, margin: 0 });
    arrow(pres, s, cx, 5.27, 0, 0.22);
    s.addText("No / log", { x: cx + 0.05, y: 5.26, w: 0.8, h: 0.22, fontFace: BODY, fontSize: 8, color: C.muted, bold: true, isTextBox: true, margin: 0 });
    box(pres, s, X0 + 0.35, 5.51, W - 0.7, 0.4, "Local DB (SQLite, offline)", "F5F5F5", "757575", 9.5);
    arrow(pres, s, cx, 5.93, 0, 0.2);
    box(pres, s, X0 + 0.35, 6.15, W - 0.7, 0.4, "Dashboard & reports", "E3F2FD", "1976D2", 9.5);
    arrow(pres, s, cx, 6.57, 0, 0.18);
    box(pres, s, X0 + 0.35, 6.77, W - 0.7, 0.38, "HQ sync (optional)", "F5F5F5", "757575", 9.5);
    s.addNotes("Kaise kaam karta hai: camera se frame lete hain (3-10 per second), YOLO11 log aur products dhoondhta hai, ByteTrack har insaan ko ek anonymous number deta hai. Phir counting, heatmap, queue aur khaali shelf nikalte hain. Koi problem mile to staff ko alert. Sab data store ke andar SQLite mein, internet aane par HQ ko sirf numbers jaate hain. Shelf model Google Colab T4 GPU par train kiya.");
  }

  // =================================================================== 4. OUTPUT & RESULTS
  if (want(4)) {
    const s = pres.addSlide();
    header(pres, s, "Prototype Output & Results");
    const imgs = [
      ["cam_entrance_live.jpg", "1. Footfall counting: live IN / OUT, people pixelated (MOT17 mall video)"],
      ["shelf_void_baseline.jpg", "2. Shelf monitoring: products (green), empty slots (red)"],
      ["heatmap_overlay_aisle.jpg", "3. Shopper heatmap: where people walk and stop"],
      ["forecast_clean.png", "4. Queue forecast on a festival rush day: error 2.8 vs 4.7 people for the baseline"],
    ];
    imgs.forEach(([f, cap], i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 0.25 + col * 4.25, y = 0.95 + row * 3.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.1, h: 2.98, rectRadius: 0.06, fill: { color: C.card }, line: { color: C.line, width: 0.75 } });
      s.addImage({ path: A(f), x: x + 0.1, y: y + 0.1, w: 3.9, h: 2.4, sizing: { type: "cover", w: 3.9, h: 2.4 } });
      s.addText(cap, { x: x + 0.1, y: y + 2.52, w: 3.9, h: 0.42, fontFace: BODY, fontSize: 10.5, bold: true, color: C.ink, valign: "middle", isTextBox: true, margin: 0 });
    });

    card(pres, s, 8.8, 0.95, 4.28, 6.13, "Key Results (unseen test data)", I.chart, { icBg: "E3EAF8" });
    const stats = [
      ["0.93", "mAP50 of our shelf detector on 2,936 unseen shelf photos", C.green],
      ["89%", "product count accuracy per shelf photo", C.green],
      ["83%", "empty-slot alerts at the right place (fixed camera)", C.blue],
      ["27%", "lower queue forecast error than the simple baseline", C.orange],
      ["64%", "footfall count accuracy on crowded street videos with human ground truth", C.purple],
      ["3 cams", "run together on one laptop CPU, no GPU", C.red],
    ];
    stats.forEach(([n, t, col], i) => {
      const y = 1.5 + i * 0.92;
      s.addText(n, { x: 8.95, y, w: 1.35, h: 0.8, fontFace: HEAD, fontSize: 24, bold: true, color: col, valign: "middle", isTextBox: true, margin: 0 });
      s.addText(t, { x: 10.35, y, w: 2.6, h: 0.8, fontFace: BODY, fontSize: 11, color: C.ink, valign: "middle", isTextBox: true, margin: 0 });
    });
    s.addNotes("Yeh sab asli output hain, fake nahi. Shelf model ko 2936 aisi photos par test kiya jo training mein nahi thi: mAP50 0.93. Footfall ko MOT17 benchmark par insaano ke label se match kiya: bheed wali sadak par 64% (store darwaze par isse behtar hoga). Queue forecast simulated data par hai - asli store ke 2-4 hafte ke data se dobara train hoga.");
  }

  // =================================================================== 5. IMPACT & BENEFITS
  if (want(5)) {
    const s = pres.addSlide();
    header(pres, s, "Impact and Benefits");
    card(pres, s, 0.25, 0.95, 6.4, 2.6, "Impact on Store Operations", I.store);
    s.addText(paragraphs([
      "Fewer empty shelves | staff get refill alerts **before shoppers notice**.",
      "Shorter queues | open a counter **15 minutes before the rush**.",
      "Better staffing | plan shifts from **real footfall and peak hours**.",
      "Smarter layout | heatmaps show **which aisles and promos work**.",
      "No manual counting | **daily and weekly reports** are automatic.",
    ], { size: 14, gap: 7 }), { x: 0.4, y: 1.43, w: 6.1, h: 2.08, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 6.8, 0.95, 6.28, 2.6, "Benefits for Shoppers & Chains", I.users, { icBg: "E3EAF8" });
    s.addText(paragraphs([
      "Products available | shoppers find what they came for.",
      "Less waiting | faster billing, **better experience**.",
      "Privacy respected | **no faces, no photos**, only numbers.",
      "Works offline | reliable in **Tier-2/3 towns** with weak internet.",
      "Multi-store view | HQ sees **all stores on one screen**.",
    ], { size: 14, gap: 7 }), { x: 6.95, y: 1.43, w: 6.0, h: 2.08, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 0.25, 3.65, 8.2, 3.6, "Economic and Strategic Gains", I.coins, { icBg: "FBF1D9" });
    s.addText(paragraphs([
      "Recover lost sales | stock-outs cost retailers **about 4% of sales** (global GMA study); faster refills win this back.",
      "Low cost | runs on **existing CCTV + a ₹35–50k mini PC**; no cloud video bills.",
      "Huge data saving | **1.2 MB/day per camera** instead of **12.6 GB** of video (**99.99% less**).",
      "Big market | India has **~13 million kirana stores** and a **US$ 1 trillion+** retail market.",
      "Future ready | fits **DPDP Act 2023** (privacy) and scales from one shop to a **national chain**.",
    ], { size: 14.5, gap: 11, hl: C.green }), { x: 0.4, y: 4.13, w: 7.9, h: 3.07, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 8.6, 3.65, 4.48, 3.6, null);
    s.addChart(pres.charts.BAR, [{ name: "Improvement %", labels: ["False shelf alerts cut", "Queue forecast error cut", "Counting accuracy gain", "Data sent vs cloud video cut"], values: [94, 27, 36, 99.99] }], {
      x: 8.7, y: 3.72, w: 4.28, h: 3.46, barDir: "bar", showTitle: true, title: "Improvements measured in our prototype (%)", titleFontSize: 11, titleColor: "111111", titleFontFace: BODY,
      chartColors: ["1E6B3A"], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: "111111", dataLabelFormatCode: "General",
      catAxisLabelFontSize: 9, catAxisLabelColor: "333333", valAxisHidden: true, valAxisMaxVal: 120, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
    });
    s.addNotes("Store ko fayda: khaali shelf jaldi bharna, rush se pehle counter kholna, staff sahi time par. Paisa: stock-out se lagbhag 4% sales jaati hai (GMA global study) - hum wapas laate hain. Kharcha kam: purane CCTV + ek chhota PC. Chart ke numbers humare apne prototype ke measurement hain, andaze nahi.");
  }

  // =================================================================== 6. FEASIBILITY & VIABILITY
  if (want(6)) {
    const s = pres.addSlide();
    header(pres, s, "Feasibility and Viability");
    card(pres, s, 0.25, 0.95, 6.95, 2.8, "Analysis of the Feasibility of the Idea", I.check, { icBg: "DDF2E3" });
    s.addText(paragraphs([
      "Proven AI | YOLO + ByteTrack are **industry standard**; our models are **trained and tested**.",
      "Hardware ready | runs on a **CPU mini PC**; Jetson / OpenVINO for more cameras.",
      "Uses existing cameras | **RTSP** from Hikvision / CP Plus / Dahua NVRs.",
      "Open datasets | **SKU-110K, COCO, MOT17**: no private data needed to start.",
      "Reliable offline | local database, **auto camera reconnect**, sync later.",
      "Easy install | **one Docker command** + zone-drawing helper tool.",
    ], { size: 13, gap: 4, hl: C.blue }), { x: 0.4, y: 1.43, w: 6.7, h: 2.28, valign: "top", isTextBox: true, margin: 0 });

    card(pres, s, 7.35, 0.95, 5.73, 2.8, null);
    s.addChart(pres.charts.BAR, [
      { name: "First version", labels: ["Footfall count accuracy", "Empty-slot alert precision", "Sparse-shelf count accuracy"], values: [47.5, 17, 55] },
      { name: "After our fixes", labels: ["Footfall count accuracy", "Empty-slot alert precision", "Sparse-shelf count accuracy"], values: [64.4, 65, 63] },
    ], {
      x: 7.45, y: 1.0, w: 5.53, h: 2.7, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Measured: before vs after our improvements (%)", titleFontSize: 11, titleFontFace: BODY, titleColor: "111111",
      chartColors: ["C0171D", "1E6B3A"], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelFormatCode: "0",
      catAxisLabelFontSize: 9, valAxisHidden: true, valAxisMaxVal: 80, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 9,
    });

    s.addText("WHAT IFs....?", { x: 0.25, y: 3.87, w: 7.3, h: 0.42, align: "center", fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, isTextBox: true, margin: 0 });
    const whatifs = [
      ["Internet goes down?", "Everything runs offline in the store; data syncs to HQ later.", "8B1A1A"],
      ["A camera fails?", "Other cameras keep working; the camera reconnects automatically.", "1E40AF"],
      ["AI makes a mistake?", "Confidence limits, alert cooldown, staff can acknowledge; recalibrate with 'Mark as Full'.", "7B4A12"],
      ["Shoppers worry about privacy?", "No faces or photos saved; anonymous IDs reset daily; clear signage.", "1B7F3A"],
      ["Store has a small budget?", "Existing CCTV + ₹35k PC; one box serves three cameras.", "6B2C91"],
    ];
    whatifs.forEach(([q, a, col], i) => {
      const x = 0.25 + i * 1.47;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.35, w: 1.39, h: 2.9, rectRadius: 0.06, fill: { color: C.card }, line: { color: col, width: 1.25 } });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.35, w: 1.39, h: 0.95, rectRadius: 0.06, fill: { color: col }, line: { color: col, width: 1.25 } });
      s.addText([{ text: "0" + (i + 1), options: { fontSize: 11, bold: true, breakLine: true } }, { text: q, options: { fontSize: 10, bold: true } }],
        { x: x + 0.05, y: 4.37, w: 1.29, h: 0.91, align: "center", valign: "middle", fontFace: BODY, color: "FFFFFF", isTextBox: true, margin: 0 });
      s.addText(a, { x: x + 0.08, y: 5.42, w: 1.25, h: 1.78, fontFace: BODY, fontSize: 13, color: C.ink, valign: "top", isTextBox: true, margin: 0 });
    });

    card(pres, s, 7.7, 3.87, 5.38, 3.38, "Strategies to Overcome Challenges", I.shield);
    s.addText(paragraphs([
      "Low computing power | light **YOLO11n**, **3–10 FPS** sampling, higher resolution only for the entrance camera.",
      "Limited Indian retail data | start with **open datasets**, then **fine-tune on store photos** using free **Colab GPU**.",
      "Crowded entrances | angled camera over the door, **960 px** input, **jitter-proof counting** (tested on MOT17).",
      "Counting bias on odd shelves | per-camera **'Mark as Full' baseline** cancels the bias.",
    ], { size: 12.5, gap: 9 }), { x: 7.85, y: 4.35, w: 5.1, h: 2.85, valign: "top", isTextBox: true, margin: 0 });
    s.addNotes("Feasible kyun hai: models already train aur test ho chuke hain, sasta hardware, purane camera. What-if: internet gaya to offline chalega, camera kharab to baaki chalte rahenge, AI galti kare to staff acknowledge/recalibrate kar sakta hai. Chart mein humare apne sudhaar: pehla version vs fix ke baad.");
  }

  // =================================================================== 7. RESEARCH & REFERENCES
  if (want(7)) {
    const s = pres.addSlide();
    header(pres, s, "Research & References");
    const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "1B7F3A" }, align: "center", valign: "middle" } });
    const rows = [
      [hdr("Feature"), hdr("Our Proposed Solution"), hdr("Existing Conventional Solutions")],
      ["Where AI runs", "Edge: inside the store, on a small PC", "Cloud servers far away"],
      ["Internet need", "Works offline; syncs numbers later", "Needs constant, fast internet"],
      ["Privacy", "No faces / photos stored; anonymous counts", "Video uploaded to the cloud"],
      ["Cost", "Existing CCTV + ₹35–50k PC", "Subscription + high bandwidth bills"],
      ["Shelf checking", "Automatic empty-slot alerts", "Manual audits / staff rounds"],
      ["Billing queues", "Predicts 15 min ahead, advises counters", "Reacts after the queue forms"],
      ["Shopper insights", "Footfall, dwell, heatmap, conversion", "Only CCTV recording"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: j === 0 ? "F1F4F2" : "FFFFFF" } } })));
    s.addTable(rows, { x: 0.25, y: 0.95, w: 7.55, colW: [1.75, 2.9, 2.9], rowH: 0.74, fontFace: BODY, fontSize: 11.5, color: C.ink, border: { type: "solid", pt: 0.75, color: "9AA5B1" }, valign: "middle" });

    card(pres, s, 7.95, 0.95, 5.13, 6.3, "Research and Dataset References", I.book, { hColor: C.red, icBg: "FBE3E4" });
    const refs = [
      ["SKU-110K", "dense retail shelf dataset (CVPR 2019)", "https://github.com/eg4000/SKU110K_CVPR19"],
      ["MOT17", "people-tracking benchmark with ground truth", "https://motchallenge.net/data/MOT17/"],
      ["COCO", "person detection dataset", "https://cocodataset.org"],
      ["ByteTrack", "multi-object tracking (ECCV 2022)", "https://arxiv.org/abs/2110.06864"],
      ["Ultralytics YOLO11", "real-time detector", "https://docs.ultralytics.com"],
      ["Out-of-Stock Detection", "Sensors 2024 (shelf voids)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825/"],
      ["Retail Out-of-Stocks", "worldwide study, GMA 2002", "https://www.supplychain247.com/images/pdfs/GMA_2002_Worldwide_OOS_Study.pdf"],
      ["Kirana modernisation", "Invest India", "https://www.investindia.gov.in/team-india-blogs/modernization-kirana-stores-india"],
      ["DPDP Act 2023", "India data-protection law", "https://www.meity.gov.in/data-protection-framework"],
      ["IBEF Retail India", "Indian retail market data", "https://www.ibef.org/industry/retail-india"],
      ["Erlang-C queueing model", "counters needed for a target wait", "https://en.wikipedia.org/wiki/Erlang_(unit)"],
    ];
    const refRuns = [];
    refs.forEach(([name, desc, url], i) => {
      refRuns.push({ text: name + " — ", options: { bold: true, bullet: true, paraSpaceAfter: 0 } });
      refRuns.push({ text: desc, options: { breakLine: true } });
      refRuns.push({ text: url, options: { hyperlink: { url }, color: "1155CC", underline: true, fontSize: 9.5, breakLine: i < refs.length - 1, paraSpaceAfter: 8 } });
    });
    s.addText(refRuns.map((r) => ({ text: r.text, options: { fontFace: BODY, fontSize: 12, color: C.ink, ...r.options } })),
      { x: 8.1, y: 1.43, w: 4.85, h: 5.75, valign: "top", isTextBox: true, margin: 0 });
    s.addNotes("Existing solutions cloud par chalte hain, internet aur paisa zyada lagta hai, video bahar jaata hai. Hum store ke andar chalte hain. References: datasets jin par train/test kiya (SKU-110K, COCO, MOT17), research papers (ByteTrack, out-of-stock detection) aur India ke data (Invest India, DPDP Act).");
  }

  await pres.writeFile({ fileName: OUT });
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
