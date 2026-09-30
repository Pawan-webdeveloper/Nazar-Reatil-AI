/* RetailSense store operations dashboard. Talks only to the local API (works offline). */
(function () {
  const $ = (s) => document.querySelector(s);
  const state = { store: null, day: null, autoDay: null, tab: "overview", cfg: null, lastTs: null };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null ? "-" : Math.round(v * 100) + "%");
  const num = (v, d = 0) => (v == null ? "-" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d }));
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const todayStr = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  async function get(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  }
  const S = () => `/api/stores/${encodeURIComponent(state.store)}`;
  const effDay = () => state.day || state.autoDay || null;
  const dayQ = () => (effDay() ? `?day=${effDay()}` : "");

  function kpi(label, value, note, color, icon, spark) {
    return `<div class="kpi t-${color}"><div class="top"><span class="ic c-${color}"><svg><use href="#${icon}"/></svg></span>
      <span class="label">${esc(label)}</span></div><div class="value">${value}</div>${note ? `<div class="note">${esc(note)}</div>` : ""}
      ${spark ? `<div class="spark" title="last 14 days">${spark}</div>` : ""}</div>`;
  }
  function statusChip(s) {
    const m = { OK: ["ok", "OK"], LOW_STOCK: ["low", "Low stock"], OUT_OF_STOCK: ["oos", "Out of stock"],
      critical: ["oos", "Critical"], warning: ["warn", "Warning"], info: ["info", "Info"] }[s] || ["info", s];
    return `<span class="status ${m[0]}"><i aria-hidden="true"></i>${esc(m[1])}</span>`;
  }
  function table(cols, rows, emptyMsg) {
    if (!rows.length) return `<div class="chart" style="height:160px" data-empty="${esc(emptyMsg || "Nothing to show for this period yet.")}"></div>`;
    return `<div class="table-wrap"><table><thead><tr>${cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>${rows
      .map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.html ? c.html(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function fillEmpty(root) { root.querySelectorAll("[data-empty]").forEach((el) => emptyState(el, el.dataset.empty)); }

  // ---------------------------------------------------------------- header
  function renderHeader() {
    const h = new Date().getHours();
    $("#greeting").textContent = h < 12 ? "Good morning," : h < 17 ? "Good afternoon," : "Good evening,";
    $("#storeName").textContent = `Here's what's happening at ${state.storeName || state.store}`;
    const d = effDay() ? new Date(effDay() + "T12:00:00") : new Date();
    $("#dateCard").textContent = d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
    $("#dateNote").textContent = state.day ? "selected day" : state.autoDay ? "latest day with data" : "today";
    const b = $("#dayBanner");
    if (!state.day && state.autoDay) {
      b.hidden = false;
      b.innerHTML = `No data for today yet, so the dashboard shows <b>${esc(state.autoDay)}</b>, the latest day with data. <button id="showToday">Show today</button>`;
      $("#showToday").onclick = () => { state.day = todayStr(); $("#daySel").value = state.day; refreshAll(); };
    } else b.hidden = true;
  }

  // ---------------------------------------------------------------- tabs
  async function loadOverview() {
    const [k, rep, series] = await Promise.all([get(`${S()}/kpis${dayQ()}`), get(`${S()}/reports/daily${dayQ()}`),
      get(`${S()}/daily_kpis?days=14${effDay() ? `&end_day=${effDay()}` : ""}`)]);
    const col = (key) => series.map((r) => r[key]);
    const alertsTotal = Object.values(k.alerts).reduce((a, b) => a + b, 0);
    $("#kpiTiles").innerHTML = [
      kpi("Footfall (entries)", num(k.footfall_in), `${num(k.footfall_out)} exits`, "blue", "i-users", sparkline(col("footfall"), css("--blue-ink"))),
      kpi("Bills (POS)", num(k.bills), k.avg_basket_value ? `avg basket ₹${num(k.avg_basket_value)}` : "POS not connected", "green", "i-cart", sparkline(col("bills"), css("--green-ink"))),
      kpi("Conversion", pct(k.conversion_rate), "bills / entries", "purple", "i-swap", sparkline(col("conversion"), css("--purple-ink"))),
      kpi("Avg queue / counter", num(k.avg_queue_len, 1), `peak ${num(k.max_queue_len, 1)} people`, "orange", "i-queue", sparkline(col("avg_queue"), css("--orange-ink"))),
      kpi("On-shelf availability", pct(k.on_shelf_availability), `${num(k.shelf_sections.OUT_OF_STOCK)} section(s) out of stock`, "teal", "i-pie", sparkline(col("shelf_fill"), css("--teal-ink"))),
      kpi("Alerts", num(alertsTotal), `${num(k.alerts.critical || 0)} critical`, "red", "i-bell", sparkline(col("alerts"), css("--red-ink"))),
    ].join("");
    const hours = Array.from({ length: 24 }, (_, h) => h);
    const byHour = Object.fromEntries(rep.hourly_footfall.map((r) => [parseInt(r.hour.slice(11, 13)), r.entries]));
    const withData = Object.keys(byHour).map(Number).filter((h) => byHour[h] > 0);
    const h0 = Math.min(7, ...withData), h1 = Math.max(22, ...withData);  // store hours, widened if data exists outside
    const open = hours.filter((h) => h >= h0 && h <= h1);
    barChart($("#chHourly"), open.map((h) => `${String(h).padStart(2, "0")}:00`), open.map((h) => byHour[h] || 0),
      { unit: "entries", empty: "Entries appear here once the entrance camera counts people crossing the line." });
    await loadDaily();
    const tr = await get(`${S()}/footfall_trend?days=28${effDay() ? `&end_day=${effDay()}` : ""}`);
    matrixChart($("#chDow"), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], hours.map((h) => String(h).padStart(2, "0")), tr.dow_hour_avg,
      { unit: "entries / hour", empty: "The weekly pattern becomes visible after a few days of entrance counts." });
  }
  async function loadDaily() {
    const days = parseInt($("#dailyRange").value, 10);
    const tr = await get(`${S()}/footfall_trend?days=${days}${effDay() ? `&end_day=${effDay()}` : ""}`);
    lineChart($("#chDaily"), tr.daily.map((d) => d.date.slice(5)), [{ name: "Entries", values: tr.daily.map((d) => d.entries), color: css("--series-1"), markers: true }],
      { empty: "Daily totals appear once the entrance camera has counted entries." });
  }

  async function loadShoppers() {
    const z = await get(`${S()}/zones${dayQ()}`);
    const rows = Object.entries(z).map(([name, v]) => ({ name, ...v }));
    $("#zoneTable").innerHTML = table([
      { label: "Zone", key: "name" }, { label: "Visits", num: true, html: (r) => num(r.visits) },
      { label: "Avg dwell (s)", num: true, html: (r) => num(r.avg_dwell_s, 1) },
      { label: "Engaged (≥10 s)", num: true, html: (r) => pct(r.engagement_rate) }], rows, "Zone visits appear once the floor camera sees shoppers.");
    fillEmpty($("#zoneTable"));
    barChart($("#chDwell"), rows.map((r) => r.name), rows.map((r) => r.avg_dwell_s || 0), { unit: "s", empty: "No zone visits for this day yet." });
    const cams = await get(`${S()}/heatmaps`);
    $("#heatmaps").innerHTML = cams.length ? cams.map((c) => `<figure><img alt="Heatmap for ${esc(c)}" src="${S()}/heatmap/${encodeURIComponent(c)}.png?t=${Date.now()}"><figcaption>${esc(c)}: blue = rarely visited, red = most visited</figcaption></figure>`).join("")
      : `<div class="chart" style="height:180px" data-empty="The heatmap builds up as shoppers move in front of the floor camera."></div>`;
    fillEmpty($("#heatmaps"));
  }

  async function loadInventory() {
    const inv = await get(`${S()}/inventory`);
    const oos = inv.filter((s) => s.status === "OUT_OF_STOCK").length, low = inv.filter((s) => s.status === "LOW_STOCK").length;
    const fill = inv.length ? inv.reduce((a, s) => a + s.fill_ratio, 0) / inv.length : null;
    $("#invTiles").innerHTML = [kpi("Sections monitored", num(inv.length), "shelf camera sections", "blue", "i-box"),
      kpi("Out of stock", num(oos), "replenish now", "red", "i-bell"), kpi("Low stock", num(low), "below threshold", "orange", "i-trend"),
      kpi("Average shelf fill", pct(fill), "facings vs capacity", "teal", "i-pie")].join("");
    $("#shelfTable").innerHTML = table([
      { label: "Section", key: "key" }, { label: "Status", html: (r) => statusChip(r.status) },
      { label: "Facings", num: true, html: (r) => num(r.facings) }, { label: "Rows", num: true, html: (r) => num(r.rows) },
      { label: "Empty slots", num: true, html: (r) => num(r.voids.length) },
      { label: "Fill", html: (r) => `<span class="bar" title="${pct(r.fill_ratio)}"><span style="width:${Math.round(r.fill_ratio * 100)}%"></span></span>${pct(r.fill_ratio)}` },
      { label: "Planogram", num: true, html: (r) => pct(r.planogram_compliance) },
      { label: "Issues", html: (r) => esc((r.issues || []).join("; ")) },
      { label: "Updated", html: (r) => new Date(r.ts * 1000).toLocaleString() },
      { label: "", html: (r) => `<button data-full="${esc(r.key)}" title="Use this snapshot as the 100 % reference after restocking">Mark as full</button>` }], inv,
      "Shelf results appear once a shelf camera sends its first snapshot.");
    fillEmpty($("#shelfTable"));
    $("#shelfTable").querySelectorAll("[data-full]").forEach((b) => b.addEventListener("click", async () => {
      const r = await fetch(`${S()}/inventory/baseline?key=${encodeURIComponent(b.dataset.full)}`, { method: "POST" });
      b.textContent = r.ok ? "Saved as 100 %" : "Failed"; b.disabled = r.ok;
    }));
    $("#replen").textContent = JSON.stringify(await get(`${S()}/replenishment`), null, 2);
  }

  async function loadQueues() {
    const q = await get(`${S()}/queue`);
    const f = q.forecast[0];
    $("#queueTiles").innerHTML = f ? [
      kpi("Queue now (total)", num(f.queue_now, 1), "people waiting", "orange", "i-queue"),
      kpi(`Forecast in ${f.horizon_min} min`, f.forecast_queue == null ? "-" : num(f.forecast_queue, 1), f.forecast_queue == null ? "forecaster warming up" : "predicted people waiting", "purple", "i-trend"),
      kpi("Counters open", num(f.counters_open), "billing counters", "blue", "i-store"),
      kpi("Recommended counters", num(f.recommended_counters), `expected wait ${f.expected_wait_min == null ? "-" : num(f.expected_wait_min, 1)} min`, "green", "i-users"),
      kpi("Avg service time", `${num(f.avg_service_min, 1)} min`, `${num(f.arrival_rate_per_min, 2)} arrivals / min`, "teal", "i-cart"),
    ].join("") : kpi("Queue", "-", "no checkout camera data yet", "orange", "i-queue");
    const byBucket = {};
    q.history_5min.forEach((r) => { byBucket[r.bucket] = (byBucket[r.bucket] || []).concat(r.value); });
    const buckets = Object.keys(byBucket).map(Number).sort((a, b) => a - b);
    lineChart($("#chQueue"), buckets.map((b) => new Date(b * 1000).toTimeString().slice(0, 5)),
      [{ name: "People waiting / counter", values: buckets.map((b) => byBucket[b].reduce((a, c) => a + c, 0) / byBucket[b].length), color: css("--series-1") }],
      { empty: "No queue measurements in the last 6 hours." });
    const live = Object.values(q.live).flatMap((v) => v.counters || []);
    $("#counterTable").innerHTML = table([
      { label: "Counter", key: "counter" }, { label: "Waiting", num: true, html: (r) => num(r.queue_length) },
      { label: "Being served", num: true, html: (r) => num(r.in_service) }, { label: "Avg wait (s)", num: true, html: (r) => num(r.avg_wait_s) },
      { label: "Avg service (s)", num: true, html: (r) => num(r.avg_service_s) }, { label: "Arrivals / min", num: true, html: (r) => num(r.arrival_rate_per_min, 2) },
      { label: "Est. wait for newcomer (s)", num: true, html: (r) => num(r.expected_wait_new_customer_s) }], live,
      "Live counter status needs the edge node running in this process.");
    fillEmpty($("#counterTable"));
  }

  async function loadAlerts() {
    const rows = await get(`${S()}/alerts?limit=200${$("#unackOnly").checked ? "&unacked=true" : ""}`);
    $("#alertTable").innerHTML = table([
      { label: "Time", html: (r) => new Date(r.ts * 1000).toLocaleString() }, { label: "Level", html: (r) => statusChip(r.level) },
      { label: "Type", key: "type" }, { label: "Where", key: "key" }, { label: "Message", key: "message" },
      { label: "", html: (r) => (r.acknowledged ? `<span class="muted">acknowledged</span>` : `<button data-ack="${r.id}">Acknowledge</button>`) }], rows,
      "No alerts. Stock shortages and queue build-ups will show up here.");
    fillEmpty($("#alertTable"));
    $("#alertTable").querySelectorAll("[data-ack]").forEach((b) => b.addEventListener("click", async () => {
      await fetch(`/api/alerts/${b.dataset.ack}/ack`, { method: "POST" }); loadAlerts(); refreshBadge();
    }));
  }

  async function loadReports() {
    const d = await get(`${S()}/reports/daily${dayQ()}`);
    $("#dailyReport").innerHTML = `<p><b>${esc(d.date)}</b>: footfall ${num(d.footfall_in)}, bills ${num(d.bills)}, conversion ${pct(d.conversion_rate)}, peak hour ${esc(d.peak_hour || "-")}, max queue ${num(d.max_queue_len, 1)}</p><ul class="recs">${d.recommendations.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    const w = await get(`${S()}/reports/weekly${effDay() ? `?end_day=${effDay()}` : ""}`);
    $("#weeklyReport").innerHTML = `<p class="muted">${esc(w.period)}; busiest day ${esc(w.busiest_day || "-")}</p>` + table([
      { label: "Day", key: "date" }, { label: "Footfall", num: true, html: (r) => num(r.footfall) }, { label: "Bills", num: true, html: (r) => num(r.bills) },
      { label: "Conversion", num: true, html: (r) => pct(r.conversion_rate) }, { label: "Avg queue", num: true, html: (r) => num(r.avg_queue_len, 1) },
      { label: "Max queue", num: true, html: (r) => num(r.max_queue_len, 1) }], w.days) + `<ul class="recs">${w.recommendations.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    const dl = (name, data, type) => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], { type })); a.download = name; a.click(); };
    $("#dlDaily").onclick = () => dl(`daily_${d.date}.json`, JSON.stringify(d, null, 2), "application/json");
    $("#dlDailyCsv").onclick = () => dl(`footfall_${d.date}.csv`, "hour,entries\n" + d.hourly_footfall.map((r) => `${r.hour},${r.entries}`).join("\n"), "text/csv");
    $("#dlWeekly").onclick = () => dl(`weekly.json`, JSON.stringify(w, null, 2), "application/json");
  }

  async function loadStores() {
    const rows = await get(`/api/stores`);
    $("#storesTable").innerHTML = table([
      { label: "Store", html: (r) => `<b>${esc(r.store_id)}</b><br><span class="muted">${esc(r.name)}</span>` },
      { label: "Status", html: (r) => (r.local ? statusChip("info").replace("Info", "This store (local)") : r.online ? statusChip("OK").replace("OK", "Online") : `<span class="status warn"><i></i>Offline, last sync ${new Date(r.last_seen * 1000).toLocaleString()}</span>`) },
      { label: "Footfall today", num: true, html: (r) => num(r.today.footfall_in) }, { label: "Bills", num: true, html: (r) => num(r.today.bills) },
      { label: "Conversion", num: true, html: (r) => pct(r.today.conversion_rate) }, { label: "Max queue", num: true, html: (r) => num(r.today.max_queue_len, 1) },
      { label: "Shelf availability", num: true, html: (r) => pct(r.today.on_shelf_availability) },
      { label: "Critical alerts", num: true, html: (r) => num((r.today.alerts || {}).critical || 0) },
      { label: "", html: (r) => `<button data-store="${esc(r.store_id)}">Open</button>` }], rows);
    $("#storesTable").querySelectorAll("[data-store]").forEach((b) => b.addEventListener("click", async () => {
      $("#storeSel").value = b.dataset.store; state.store = b.dataset.store; await pickAutoDay(); switchTab("overview");
    }));
  }

  let liveTimer = null;
  async function loadLive() {
    const cams = await get(`/api/cameras`);
    $("#liveGrid").innerHTML = cams.map((c) => `<figure><img alt="Live view ${esc(c.id)}" data-cam="${esc(c.id)}" src="/api/cameras/${encodeURIComponent(c.id)}/preview.jpg?t=${Date.now()}" onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'muted',textContent:'No live feed for ${esc(c.id)}: start the edge node with python run.py edge'}))"><figcaption>${esc(c.id)} · ${esc(c.role)}${c.live ? ` · ${num(c.live.fps, 1)} FPS` : ""}</figcaption></figure>`).join("");
    clearInterval(liveTimer);
    liveTimer = setInterval(() => {
      if (state.tab !== "live") return clearInterval(liveTimer);
      document.querySelectorAll("#liveGrid img[data-cam]").forEach((img) => { img.src = `/api/cameras/${encodeURIComponent(img.dataset.cam)}/preview.jpg?t=${Date.now()}`; });
    }, 1000);
  }

  const loaders = { overview: loadOverview, shoppers: loadShoppers, inventory: loadInventory, queues: loadQueues, alerts: loadAlerts, reports: loadReports, stores: loadStores, live: loadLive };
  async function switchTab(tab) {
    state.tab = tab;
    document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${tab}`));
    $("#sidebar").classList.remove("open");
    renderHeader();
    try { await loaders[tab](); } catch (e) { console.error(e); }
    try { localStorage.setItem("rs_tab", tab); } catch (_) {}
  }
  const refreshAll = () => switchTab(state.tab);
  async function refreshBadge() {
    try { const a = await get(`${S()}/alerts?limit=500&unacked=true`); $("#alertCount").textContent = a.length ? a.length : ""; } catch (_) {}
  }
  async function refreshHealth() {
    try {
      const [h, ld] = await Promise.all([get(`/api/health`), get(`${S()}/latest_day`)]);
      const cams = h.edge ? h.edge.cameras : [];
      const live = cams.filter((c) => c.alive).length;
      $("#edgeText").textContent = h.edge ? `edge: ${live}/${cams.length} cameras live` : "edge: dashboard only";
      $("#edgeStatus .dot").classList.toggle("off", !h.edge || !live);
      $("#sysDot").classList.toggle("off", !h.edge || !live);
      $("#sysTitle").textContent = h.edge && live ? "System online" : "Dashboard online";
      const ago = ld.ts ? Math.max(0, Math.round((Date.now() / 1000 - ld.ts) / 60)) : null;
      $("#sysSub").textContent = ago == null ? "no data received yet" : ago < 1 ? "last data just now" : ago < 120 ? `last data ${ago} min ago` : `last data ${Math.round(ago / 60)} h ago`;
    } catch (_) { $("#edgeText").textContent = "edge: unreachable"; $("#sysTitle").textContent = "Server unreachable"; }
  }
  async function pickAutoDay() {
    state.autoDay = null;
    if (state.day) return;
    try {
      const ld = await get(`${S()}/latest_day`);
      if (ld.day && ld.day !== todayStr()) state.autoDay = ld.day;
    } catch (_) {}
  }

  async function init() {
    // URL overrides (shareable links / screenshots): ?theme=light|dark&tab=inventory&day=YYYY-MM-DD
    const qp = new URLSearchParams(location.search);
    let saved = qp.get("theme");
    try { saved = saved || localStorage.getItem("rs_theme"); } catch (_) {}
    if (saved) document.documentElement.dataset.theme = saved;
    if (qp.get("day")) { state.day = qp.get("day"); $("#daySel").value = state.day; }
    $("#themeBtn").onclick = () => {
      const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("rs_theme", document.documentElement.dataset.theme); } catch (_) {}
      refreshAll();
    };
    $("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");
    state.cfg = await get(`/api/config`);
    const stores = await get(`/api/stores`);
    const names = Object.fromEntries(stores.map((s) => [s.store_id, s.name]));
    const ids = stores.map((s) => s.store_id);
    if (!ids.includes(state.cfg.store_id)) ids.unshift(state.cfg.store_id);
    $("#storeSel").innerHTML = ids.map((i) => `<option>${esc(i)}</option>`).join("");
    state.store = state.cfg.store_id;
    state.storeName = state.cfg.store_name;
    $("#storeSel").value = state.store;
    $("#storeSel").onchange = async (e) => { state.store = e.target.value; state.storeName = names[state.store] || state.store; await pickAutoDay(); refreshAll(); refreshBadge(); };
    $("#daySel").onchange = async (e) => { state.day = e.target.value || null; await pickAutoDay(); refreshAll(); };
    $("#dailyRange").onchange = () => loadDaily().catch(console.error);
    $("#unackOnly").onchange = loadAlerts;
    document.querySelectorAll(".nav button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
    await pickAutoDay();
    let tab = qp.get("tab") || "overview";
    try { tab = qp.get("tab") || localStorage.getItem("rs_tab") || "overview"; } catch (_) {}
    await switchTab(loaders[tab] ? tab : "overview");
    refreshBadge(); refreshHealth();
    setInterval(() => { refreshBadge(); refreshHealth(); if (["overview", "queues", "inventory"].includes(state.tab)) loaders[state.tab]().catch(() => {}); }, 30000);
    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(refreshAll, 200); });
  }
  init().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<div class="card">Could not reach the API: ${esc(e.message)}</div>`); });
})();
