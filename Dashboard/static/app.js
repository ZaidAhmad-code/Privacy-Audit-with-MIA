const $ = s => document.querySelector(s);
const api = p => fetch(p).then(r => r.json());
const f = (x, d = 3) => Number(x).toFixed(d);
const band = s => (s < 5 ? "ok" : s < 15 ? "warn" : "bad");
let summary = [], models = [], selected = "baseline_e50", seed = 1, tok = 0, pinned = null, last = null, di = -1, view = "mia", lsum = [], lmodels = [];
const cur = () => (view === "lira" ? lsum : summary);

const root = document.documentElement;
function setTheme(t) {
  root.dataset.theme = t;
  $("#theme").textContent = t === "dark" ? "Light mode" : "Dark mode";
}
setTheme(matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
$("#theme").onclick = () => setTheme(root.dataset.theme === "dark" ? "light" : "dark");

const nm = id => (id === "baseline_e50" ? "Overfit baseline" : id === "baseline_early" ? "Early-stopped" : id.startsWith("baseline_wd") ? "Weight decay λ=" + id.replace("baseline_wd", "") : id.replace("dpsgd_eps", "DP-SGD ε "));

function side() {
  const baselines = models.filter(m => m.kind !== "dp");
  const dp = models.filter(m => m.kind === "dp");
  const btn = m => {
    const d = cur().find(x => x.model === m.id);
    return `<button class="mi${m.id === selected ? " on" : ""}${view === "lira" && !lmodels.includes(m.id) ? " off" : ""}" data-id="${m.id}"><i class="dot ${d ? band(d.risk_score) : "ok"}"></i><span>${nm(m.id)}</span></button>`;
  };
  $("#models").innerHTML =
    `<p class="cap" style="margin:6px 8px 6px">Baselines & Defences</p>` +
    baselines.map(btn).join("") +
    `<p class="cap" style="margin:16px 8px 6px">DP-SGD (Provable)</p>` +
    dp.map(btn).join("");
  $("#models").querySelectorAll("button").forEach(b => (b.onclick = () => { if (b.classList.contains("off")) return; selected = b.dataset.id; run(); }));
  $("#seeds").innerHTML = [1, 2, 3, 4, 5, "avg"].map(s => `<button class="${s === seed ? "on" : ""}" data-s="${s}">${s === "avg" ? "Avg" : s}</button>`).join("");
  $("#seeds").querySelectorAll("button").forEach(b => (b.onclick = () => { seed = b.dataset.s === "avg" ? "avg" : +b.dataset.s; run(); }));
}

function spark(v, c) {
  const w = 160, h = 40, lo = Math.min(...v), r = Math.max(...v) - lo || 1;
  const p = v.map((x, i) => `${(i / (v.length - 1)) * w},${h - 4 - ((x - lo) / r) * (h - 8)}`).join(" ");
  return `<svg viewBox="0 0 ${w} ${h}" class="sp" preserveAspectRatio="none"><polygon points="0,${h} ${p} ${w},${h}" fill="${c}" opacity=".13"/><polyline points="${p}" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
}

function kpis() {
  const S = cur(), g = id => S.find(d => d.model === id) || {};
  const dp = S.filter(d => d.kind === "dp"), base = g("baseline_e50"), best = Math.max(...dp.map(d => d.test_auc));
  const t = (l, v, sub, vals, c) => `<div class="kpi"><span class="kl">${l}</span><b class="kv">${v}</b><span class="ks">${sub}</span>${spark(vals, c)}</div>`;
  const last4 = view === "lira"
    ? t("Early-stopped risk", f(g("baseline_early").risk_score, 1), "still leaks a little", S.map(d => d.risk_score), "var(--warn)")
    : t("Cost of privacy", f(best - g("dpsgd_eps0.5").test_auc), "lost at ε = 0.5", dp.map(d => d.test_auc), "var(--warn)");
  $("#kpis").innerHTML =
    t("Overfit model risk", f(base.risk_score, 1), "out of 100", S.map(d => d.risk_score), "var(--bad)") +
    t("Worst DP risk", f(Math.max(...dp.map(d => d.risk_score)), 1), view === "lira" ? "worst of the ε levels tested" : "worst of six ε levels", dp.map(d => d.risk_score), "var(--ok)") +
    t("Best DP usefulness", f(best), `overfit: ${f(base.test_auc)}`, S.map(d => d.test_auc), "var(--acc)") + last4;
}

function gauge(score) {
  const arc = "M40.7 150A80 80 0 1 1 179.3 150", len = 335.1, frac = Math.min(score, 100) / 100;
  return `<svg viewBox="0 0 220 190" class="gauge" role="img" aria-label="Privacy risk score ${score} out of 100"><path d="${arc}" class="gt"/><path d="${arc}" class="gv" stroke-dasharray="${len * frac} ${len}"/><text x="110" y="112" class="gn">${score}</text><text x="110" y="136" class="gs">risk score</text></svg>`;
}

function explain(r) {
  const b = band(r.risk_score);
  if (b === "ok") return "The attacker was just guessing.";
  if (b === "warn") return "The attacker did a bit better than guessing.";
  return "The attacker could often tell who was in the training data.";
}

const stat = (n, v, note) => `<div><dt>${n}</dt><dd class="v">${v}</dd><dd class="n">${note}</dd></div>`;

const stats = r => (r.attack === "lira"
  ? stat("Attack score", f(r.attack_auc), "0.5 = guessing") + stat("Usefulness", f(r.test_auc), "on unseen patients") +
    stat("Caught at 1% false alarms", pct(r.catch1), "no leak baseline: about 1.3%") + stat("Caught at 0.1%", pct(r.catch01), "no leak baseline: about 0.2%")
  : stat("Attack score", f(r.attack_auc), "0.5 = guessing") + stat("Usefulness", f(r.test_auc), "on unseen patients") +
    stat("Attacker accuracy", f(r.attack_acc), "0.5 = guessing") + stat("Memorisation gap", f(r.gap), "train vs test accuracy"));

function guarantee(r) {
  if (r.model.startsWith("dpsgd")) {
    const eps = r.model.replace("dpsgd_eps", "ε = ");
    return `<span class="guar-tag dp">🛡️ Provable DP (${eps})</span>`;
  }
  if (r.model.startsWith("baseline_wd")) {
    const wd = r.model.replace("baseline_wd", "λ = ");
    return `<span class="guar-tag reg">⚖️ L2 Regularisation (${wd})</span>`;
  }
  if (r.model === "baseline_early") {
    return `<span class="guar-tag warn">⏱️ Heuristic (Early Stopping)</span>`;
  }
  return `<span class="guar-tag bad">⚠️ No Defence (Overfit)</span>`;
}

function renderHero(r) {
  const b = band(r.risk_score);
  $("#result").innerHTML = `<div class="hero ${b}"><div>${gauge(r.risk_score)}</div><div class="vd">
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span class="pill">${r.band}</span>${guarantee(r)}</div>
    <h2>${explain(r)}</h2>
    <p class="meta">${r.label} · ${r.seed === "avg" ? "average of 5 runs" : "run " + r.seed}</p>
    <dl class="stats">${stats(r)}</dl><button class="pin" id="pin" type="button">Pin to compare</button></div></div>`;
}

function render(r) {
  last = r;
  renderHero(r);
  const same = pinned && pinned.model === r.model && pinned.seed === r.seed;
  $("#pin").textContent = same ? "Pinned" : "Pin to compare";
  $("#pin").onclick = () => { pinned = r; render(r); };
  curves(r);
}

function dist(r) {
  const W = 420, H = 200, m = { l: 8, r: 8, t: 10, b: 28 }, a = r.hist.members, b = r.hist.others, lira = r.attack === "lira";
  const mx = Math.max(...a, ...b, 0.0001), bw = (W - m.l - m.r) / a.length, y = v => H - m.b - (v / mx) * (H - m.t - m.b);
  const bars = (arr, c) => arr.map((v, i) => `<rect x="${m.l + i * bw + 1}" y="${y(v)}" width="${bw - 2}" height="${H - m.b - y(v)}" rx="2" fill="${c}" opacity=".6"/>`).join("");
  const n = v => (lira ? (v * 100).toFixed(0) + "%" : Math.round(v));
  const tip = i => {
    const lo = r.hist.edges ? r.hist.edges[i] : i / a.length, hi = r.hist.edges ? r.hist.edges[i + 1] : (i + 1) / a.length;
    return `Score ${lo.toFixed(2)} to ${hi.toFixed(2)}<br>${n(a[i])} ${lira ? "of training patients" : "training patients"}<br>${n(b[i])} ${lira ? "of unseen patients" : "unseen patients"}`;
  };
  const hits = a.map((_, i) => `<rect class="hit" x="${m.l + i * bw}" y="0" width="${bw}" height="${H - m.b}" data-tip="${tip(i)}"/>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" class="mini">${bars(b, "var(--acc)")}${bars(a, "var(--bad)")}<line class="gl" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><text class="tk" x="${m.l}" y="${H - 9}">looks unseen</text><text class="tk" x="${W - m.r}" y="${H - 9}" text-anchor="end">looks trained on</text>${hits}</svg>`;
}

function roc(r, p) {
  const S = 220, m = 34, sc = S - m - 10, lira = r.attack === "lira";
  const L = v => (lira ? Math.log10(Math.max(v, 0.001)) / 3 + 1 : v), X = v => m + L(v) * sc, Y = v => S - m - L(v) * sc;
  const path = c => c.fpr.map((x, i) => `${i ? "L" : "M"}${X(x)} ${Y(c.tpr[i])}`).join("");
  const lab = v => `${+(v * 100).toFixed(1)}%`;
  const ticks = lira ? [0.001, 0.01, 0.1, 1].map(v => `<line class="gl" x1="${X(v)}" x2="${X(v)}" y1="${Y(1)}" y2="${Y(0)}"/><line class="gl" x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tk" x="${X(v)}" y="${S - m + 13}" text-anchor="middle">${lab(v)}</text><text class="tk" x="${m - 5}" y="${Y(v) + 4}" text-anchor="end">${lab(v)}</text>`).join("") : "";
  const hits = r.roc.fpr.map((x, i) => `<circle class="hit" cx="${X(x)}" cy="${Y(r.roc.tpr[i])}" r="7" data-tip="At ${pct(x)} false alarms,<br>it catches ${pct(r.roc.tpr[i])} of training patients"/>`).join("");
  return `<svg viewBox="0 0 ${S} ${S}" class="mini sq">${ticks}<line class="hv" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>
    <path d="${path(r.roc)}" fill="none" stroke="var(--bad)" stroke-width="2.2" stroke-linejoin="round"/>
    ${p ? `<path d="${path(p.roc)}" fill="none" stroke="var(--acc)" stroke-width="2.2" stroke-dasharray="5 4" stroke-linejoin="round"/>` : ""}
    <text class="tk" x="${(m + S) / 2}" y="${S - 3}" text-anchor="middle">false alarms</text><text class="tk" transform="translate(9 ${S / 2 - 10}) rotate(-90)" text-anchor="middle">patients caught</text>${hits}</svg>`;
}

function curves(r) {
  const cmp = pinned && !(pinned.model === r.model && pinned.seed === r.seed);
  const row = x => `<div><b>${nm(x.model)} · ${x.seed === "avg" ? "average" : "run " + x.seed}</b><span>Risk ${f(x.risk_score, 1)} · usefulness ${f(x.test_auc)} · gap ${f(x.gap)}</span></div>`;
  $("#curves").innerHTML = (cmp ? `<div class="cmp">${row(pinned)}<em>vs</em>${row(r)}<button id="unpin" type="button">Clear</button></div>` : "") +
    `<div class="cards"><div class="card"><h3>${r.attack === "lira" ? "How LiRA scores patients" : "How the attack scores patients"}</h3><p class="sub">Shapes far apart mean a leak. Shapes on top of each other mean guessing.</p>${dist(r)}
      <p class="lg2"><span><i style="background:var(--bad)"></i>Training patients</span><span><i style="background:var(--acc)"></i>Unseen patients</span></p></div>
     <div class="card"><h3>${r.attack === "lira" ? "ROC curve, log scale" : "ROC curve"}</h3><p class="sub">${r.attack === "lira" ? "Zoomed in on few false alarms. " : ""}Closer to the dotted line means guessing.${cmp ? " Dashed blue is the pinned model." : ""}</p>${roc(r, cmp ? pinned : null)}</div></div>`;
  const u = $("#unpin");
  if (u) u.onclick = () => { pinned = null; render(last); };
}

const VSUB = {
  mia: "The attack from the proposal. A practice model learns what training patients look like, then checks the real model.",
  lira: "A stronger attack. Many practice models show how each patient is treated with and without being in the training data.",
};
function syncView() {
  document.querySelectorAll("#views button").forEach(b => b.classList.toggle("on", b.dataset.v === view));
  $("#vsub").textContent = VSUB[view];
  if (view === "lira" && !lmodels.includes(selected)) selected = "baseline_e50";
  kpis();
}
function setView(v) { view = v; pinned = null; syncView(); run(); }
document.querySelectorAll("#views button").forEach(b => (b.onclick = () => setView(b.dataset.v)));

const gt = document.createElement("div");
gt.id = "gtip"; gt.className = "tip"; gt.hidden = true;
document.body.append(gt);
document.addEventListener("mousemove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { gt.hidden = true; return; }
  gt.hidden = false; gt.innerHTML = t.dataset.tip; gt.style.left = e.clientX + "px"; gt.style.top = e.clientY + "px";
});

const steps = [
  { m: "baseline_e50", t: "An overfit model. The attacker can tell who was in its training data." },
  { m: "dpsgd_eps1", pinModel: "baseline_e50", t: "Same data, trained with DP-SGD. Now the attacker is just guessing." },
  { t: "Side by side: the leak is gone.", scroll: "#curves" },
  { t: "The price is a small drop in usefulness.", scroll: "#cp" },
];

async function showStep(i) {
  const s = steps[i];
  if (!s) return exitDemo();
  di = i;
  $("#demo").hidden = false;
  $("#dt").textContent = s.t;
  $("#dn").textContent = i === steps.length - 1 ? "Done" : "Next";
  if (s.pinModel) pinned = await api(`/api/audit?model=${s.pinModel}&seed=1`);
  if (s.m) { selected = s.m; seed = 1; await run(); }
  if (s.scroll) document.querySelector(s.scroll).scrollIntoView({ behavior: "smooth", block: "center" });
  else window.scrollTo({ top: 0, behavior: "smooth" });
}
function exitDemo() { di = -1; $("#demo").hidden = true; }
$("#demoBtn").onclick = () => { pinned = null; view = "mia"; syncView(); showStep(0); };
$("#dn").onclick = () => showStep(di + 1);
$("#db").onclick = () => showStep(Math.max(di - 1, 0));
$("#dx").onclick = exitDemo;

const short = d => (
  d.kind === "baseline" ? "Overfit" :
  d.kind === "reference" ? "Early stop" :
  d.kind === "regularised" ? "WD " + d.model.replace("baseline_wd", "") :
  d.model.replace("dpsgd_eps", "ε ")
);

function chart() {
  const S = cur();
  if (!S.length) return;
  const W = 760, H = 310, m = { l: 48, r: 52, t: 16, b: 42 }, n = S.length, cw = (W - m.l - m.r) / (n - 1);
  const X = i => m.l + i * cw, rmax = Math.max(30, ...S.map(d => d.risk_score)) * 1.1, u0 = 0.5, u1 = 0.66;
  const YR = v => H - m.b - (v / rmax) * (H - m.t - m.b), YU = v => H - m.b - ((v - u0) / (u1 - u0)) * (H - m.t - m.b);
  const pu = S.map((d, i) => `${X(i)},${YU(d.test_auc)}`).join(" "), pr = S.map((d, i) => `${X(i)},${YR(d.risk_score)}`).join(" ");
  let s = `<svg viewBox="0 0 ${W} ${H}" id="cs"><defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--acc)" stop-opacity=".28"/><stop offset="1" stop-color="var(--acc)" stop-opacity="0"/></linearGradient></defs>`;
  for (let i = 0; i <= 4; i++) {
    const y = m.t + ((H - m.t - m.b) * i) / 4;
    s += `<line class="gl" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/><text class="tk" x="${m.l - 10}" y="${y + 4}" text-anchor="end">${(rmax * (1 - i / 4)).toFixed(0)}</text><text class="tk" x="${W - m.r + 10}" y="${y + 4}">${(u1 - ((u1 - u0) * i) / 4).toFixed(2)}</text>`;
  }
  s += `<polygon points="${X(0)},${H - m.b} ${pu} ${X(n - 1)},${H - m.b}" fill="url(#g)"/><polyline class="ln" points="${pu}" stroke="var(--acc)"/><polyline class="ln" points="${pr}" stroke="var(--bad)"/><line id="hv" class="hv" y1="${m.t}" y2="${H - m.b}" x1="-9" x2="-9"/>`;
  S.forEach((d, i) => {
    const on = d.model === selected, r = on ? 6 : 3.5, c = on ? "sel" : "";
    s += `<circle cx="${X(i)}" cy="${YU(d.test_auc)}" r="${r}" fill="var(--acc)" class="${c}"/><circle cx="${X(i)}" cy="${YR(d.risk_score)}" r="${r}" fill="var(--bad)" class="${c}"/><text class="tk${on ? " on" : ""}" x="${X(i)}" y="${H - 14}" text-anchor="middle">${short(d)}</text><rect class="hit" data-i="${i}" x="${X(i) - cw / 2}" y="0" width="${cw}" height="${H}"/>`;
  });
  $("#chart").innerHTML = s + `</svg><div id="tip" class="tip" hidden></div>`;
  const tip = $("#tip"), hv = $("#hv");
  $("#cs").querySelectorAll(".hit").forEach(h => {
    const i = +h.dataset.i, d = S[i];
    h.onmouseenter = () => {
      hv.setAttribute("x1", X(i)); hv.setAttribute("x2", X(i));
      tip.hidden = false; tip.style.left = (X(i) / W) * 100 + "%"; tip.style.top = (Math.min(YR(d.risk_score), YU(d.test_auc)) / H) * 100 + "%";
      tip.innerHTML = `<b>${d.label}</b><span><i style="background:var(--bad)"></i>Privacy risk ${f(d.risk_score, 1)}</span><span><i style="background:var(--acc)"></i>Usefulness ${f(d.test_auc)}</span>`;
    };
    h.onclick = () => { selected = d.model; run(); };
  });
  $("#cs").onmouseleave = () => { tip.hidden = true; hv.setAttribute("x1", -9); hv.setAttribute("x2", -9); };
}

async function run() {
  const my = ++tok;
  side(); chart();
  $("#result").classList.add("busy");
  try {
    const r = await api(`/api/${view === "lira" ? "lira" : "audit"}?model=${selected}&seed=${seed}`);
    if (my !== tok) return;
    if (r.error) throw new Error(r.error);
    render(r);
  } catch (e) {
    if (my === tok) $("#result").innerHTML = `<p class="err">Audit failed: ${e.message}</p>`;
  }
  $("#result").classList.remove("busy");
}

(async () => {
  try {
    const mj = await api("/api/models?all=1");
    models = mj.models; lmodels = mj.lira || [];
    summary = await api("/api/summary?all=1");
    const ex = await api("/api/extras"), ordr = id => models.findIndex(x => x.id === id);
    lsum = (Array.isArray(ex.lira) ? ex.lira : []).filter(r => lmodels.includes(r.model)).map(r => {
      const q = summary.find(d => d.model === r.model) || {};
      return { model: r.model, label: q.label, kind: q.kind, test_auc: q.test_auc, test_auc_sd: q.test_auc_sd || 0,
               risk_score: Math.round(Math.max(0, (r.auc - 0.5) / 0.5 * 1000)) / 10, risk_score_sd: Math.round(r.auc_sd / 0.5 * 1000) / 10 };
    }).sort((x, y) => ordr(x.model) - ordr(y.model));
    syncView(); run();
  } catch (e) {
    $("#result").innerHTML = `<p class="err">Can’t reach the server. Start it with python dashboard/app.py.</p>`;
  }
})();

// Top-Level Tab Switching
document.querySelectorAll("#mainTabs .tab-btn").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#mainTabs .tab-btn").forEach(b => b.classList.remove("on"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("on");
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add("active");
  };
});

/* ---- other tests, shown as cards ---- */
const pct = x => (x * 100).toFixed(1) + "%";
const sg = x => (x >= 0 ? "+" : "") + x.toFixed(3);
const fail = x => `<p class="err">${x.error}</p>`;
const card = (t, sub, body, cat = "") => `<div class="card"${cat ? ` data-cat="${cat}"` : ""}><h3>${t}</h3><p class="sub">${sub}</p>${body}</div>`;

async function findings() {
  const [m, e] = await Promise.all([api("/api/models?all=1"), api("/api/extras")]);
  const ord = id => m.models.findIndex(x => x.id === id), by = (a, b) => ord(a.model) - ord(b.model);
  const kc = id => (id === "baseline_e50" ? "var(--bad)" : id === "baseline_early" ? "var(--warn)" : "var(--ok)");

  // 1. Simple attacks
  let simple;
  if (e.simple.error) simple = fail(e.simple);
  else {
    const p = {};
    e.simple.forEach(r => ((p[r.model] ||= { model: r.model })[r.attack] = r.auc));
    simple = `<div class="heat"><span class="h"></span><span class="h">Loss</span><span class="h">Confidence</span><span class="h">Label only</span>` +
      Object.values(p).sort(by).map(r => `<span class="n">${nm(r.model)}</span>` +
        ["loss", "confidence", "label_only"].map(k => `<span class="cell" style="--a:${Math.min(Math.max((r[k] - 0.5) / 0.1, 0), 1)}">${r[k].toFixed(2)}</span>`).join("")).join("") + `</div>`;
  }

  // 2. Exposure analysis
  const x = e.exposure;
  const exposed = x.error ? fail(x) :
    `<div class="row"><span>Most exposed</span><span class="bar"><b style="width:${x.readmitted_share_top * 100}%"></b></span><em>${pct(x.readmitted_share_top)}</em></div>
     <div class="row"><span>Everyone</span><span class="bar"><b style="width:${x.readmitted_share_all * 100}%;--c:var(--mute)"></b></span><em>${pct(x.readmitted_share_all)}</em></div>
     <p class="cap2">What else stands out (in standard deviations)</p>` +
    x.features.slice(0, 5).map(d => `<div class="row w"><span>${d.feature.replace(/_/g, " ")}</span><span class="bar"><b style="width:${Math.min(Math.abs(d.std_diff) / 0.6, 1) * 100}%;--c:${d.std_diff < 0 ? "var(--ok)" : "var(--bad)"}"></b></span><em>${d.std_diff > 0 ? "+" : ""}${d.std_diff.toFixed(2)}</em></div>`).join("");

  // 3. Attribute inference
  const attr = e.attribute.error ? fail(e.attribute) :
    `<ul class="plain">${e.attribute.sort(by).map(r => `<li><span>${nm(r.model)}</span><em>${sg(r.memorisation)}</em></li>`).join("")}</ul><p class="note-s">Extra success on training patients. Zero means no leak.</p>`;

  // 4. Patient vs visit leakage
  let patCard = "";
  if (e.patient_level && !e.patient_level.error) {
    const list = e.patient_level.sort(by).map(r => {
      const c = r.several_visits > 0.7 ? "var(--bad)" : r.several_visits > 0.55 ? "var(--warn)" : "var(--ok)";
      return `<div class="row"><span>${nm(r.model)}</span><span class="bar"><b style="width:${Math.min(Math.max(0, (r.several_visits - 0.48) / 0.35) * 100, 100)}%;--c:${c}"></b></span><em>${r.several_visits.toFixed(3)}</em></div>`;
    }).join("");
    patCard = card("Patient vs visit leakage", "Patients with multiple visits leak heavily in overfit models (AUC 0.804 vs 0.583 for 1 visit). DP-SGD keeps both safe (~0.52).",
      `<div class="cap2">Attack AUC on multi-visit patients</div>${list}`, "attacks");
  }

  // 5. LiRA vs chance baseline (Null check)
  let nullCard = "";
  if (e.null_check && !e.null_check.error) {
    const rows = e.null_check.filter(r => r.real_auc != null).sort((a,b) => ord(a.model) - ord(b.model)).map(r => {
      const diff = r.extra_auc;
      const b = diff > 0.1 ? "bad" : diff > 0.02 ? "warn" : "ok";
      return `<tr><td><b>${nm(r.model)}</b></td><td class="r">${r.real_auc.toFixed(3)}</td><td class="r">${r.null_auc.toFixed(3)}</td><td class="r"><span class="badge ${b}">${diff >= 0 ? "+" : ""}${diff.toFixed(3)}</span></td></tr>`;
    }).join("");
    nullCard = card("LiRA vs chance baseline", "Chance on models that never saw these patients is ~0.50 AUC (1.3% caught). Only scores above chance count as a real leak.",
      `<table class="tbl"><thead><tr><th>Model</th><th class="r">Real</th><th class="r">Chance</th><th class="r">Extra AUC</th></tr></thead><tbody>${rows}</tbody></table>`, "ablations");
  }

  // 6. Weight decay as defense
  let wdCard = "";
  if (e.weight_decay && !e.weight_decay.error) {
    const rows = e.weight_decay.map(r => {
      const b = r.lira_auc > 0.6 ? "bad" : r.lira_auc > 0.515 ? "warn" : "ok";
      const lbl = r.model.replace("baseline_wd", "Weight decay λ=");
      return `<tr><td><b>${lbl}</b></td><td class="r">${r.test_auc.toFixed(3)}</td><td class="r">${r.lira_auc.toFixed(3)}</td><td class="r"><span class="badge ${b}">${(r.lira_tpr1 * 100).toFixed(1)}%</span></td></tr>`;
    }).join("");
    wdCard = card("Regularisation defence (Weight decay)", "L2 penalty on plain models. λ=0.01 matches early-stopping utility (AUC 0.621) with low leak, but lacks provable DP guarantees.",
      `<table class="tbl"><thead><tr><th>Setting</th><th class="r">Test AUC</th><th class="r">LiRA AUC</th><th class="r">Caught@1%</th></tr></thead><tbody>${rows}</tbody></table>`, "ablations");
  }

  // 7. Probability recalibration (Fixing DP)
  let calCard = "";
  if (e.calibration && !e.calibration.error) {
    const rows = e.calibration.filter(r => ["baseline_e50", "baseline_early", "dpsgd_eps1", "dpsgd_eps10"].includes(r.model)).map(r => {
      return `<tr><td><b>${nm(r.model)}</b></td><td class="r">${r.ece_before.toFixed(3)}</td><td class="r"><b style="color:var(--ok)">${r.ece_after.toFixed(3)}</b></td><td class="r">${(r.mean_prob_after * 100).toFixed(1)}%</td></tr>`;
    }).join("");
    calCard = card("Probability recalibration (Fixing DP)", "DP gradient clipping depresses raw predictions to ~1.4%. A validation logistic calibration drops error (ECE) 10x without hurting AUC.",
      `<table class="tbl"><thead><tr><th>Model</th><th class="r">Raw ECE</th><th class="r">Calibrated</th><th class="r">Mean Prob</th></tr></thead><tbody>${rows}</tbody></table><p class="note-s" style="margin-top:10px">Base readmission rate is 11.2%. Calibrated models restore correct probabilities.</p>`, "dp");
  }

  // 8. Gradient clipping sweep (ε=10)
  let clipCard = "";
  if (e.clipping && !e.clipping.error) {
    const rows = e.clipping.map(r => {
      return `<tr><td><b>C = ${r.clip}</b></td><td class="r">${(r.mean_prob * 100).toFixed(1)}%</td><td class="r">${r.test_auc.toFixed(3)}</td><td class="r">${r.ece.toFixed(3)}</td></tr>`;
    }).join("");
    clipCard = card("Gradient clipping sweep (ε = 10)", "Testing clipping norm C from 0.1 to 10.0. Higher C restores predicted probabilities to 10.7%, proving clipping causes the shift.",
      `<table class="tbl"><thead><tr><th>Clip Norm</th><th class="r">Mean Prob</th><th class="r">Test AUC</th><th class="r">ECE Error</th></tr></thead><tbody>${rows}</tbody></table>`, "dp");
  }

  // 9. Clinical Decision Thresholds (Top 20%)
  let thCard = "";
  if (e.thresholds && !e.thresholds.error) {
    const rows = e.thresholds.sort(by).map(r => {
      return `<tr><td><b>${nm(r.model)}</b></td><td class="r">${(r.precision * 100).toFixed(1)}%</td><td class="r">${(r.recall * 100).toFixed(1)}%</td><td class="r"><span class="badge ok">${r.lift}x</span></td></tr>`;
    }).join("");
    thCard = card("Clinical screening: Top 20% highest risk", "Precision, recall, and lift on unseen patients when flagging the top 20% predicted risk (random base rate is 11.1%).",
      `<table class="tbl"><thead><tr><th>Model</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">Lift</th></tr></thead><tbody>${rows}</tbody></table>`, "ablations");
  }

  // 10. DP-SGD Privacy Accounting
  let accCard = "";
  if (e.accounting && !e.accounting.error) {
    const rows = e.accounting.map(r => {
      return `<tr><td><b>ε = ${r.target_epsilon}</b></td><td class="r">${r.spent_epsilon.toFixed(3)}</td><td class="r">${r.noise_multiplier.toFixed(2)}</td><td class="r">10⁻⁵</td></tr>`;
    }).join("");
    accCard = card("Privacy accounting (Rényi DP)", "Strict Opacus accounting over 2,350 steps (50 epochs) on 5,969 training visits with sampling rate 0.0213 and clip norm 1.0.",
      `<table class="tbl"><thead><tr><th>Target ε</th><th class="r">Spent ε</th><th class="r">Noise (σ)</th><th class="r">Delta (δ)</th></tr></thead><tbody>${rows}</tbody></table>`, "dp");
  }

  const elF = $("#cards-forensics");
  if (elF) {
    elF.innerHTML =
      patCard +
      card("Who is most exposed", "Overfit model, top 5% most exposed patients. Share who were readmitted.", exposed) +
      card("Hidden detail test (Attribute inference)", "Can the attacker guess a patient’s gender from the model? It works the same on patients the model never saw, so it isn’t leaking training data.", attr) +
      card("Simple attacks matrix", "Attack score for three basic attacks across models. 0.5 is guessing.", simple);
  }

  const elA = $("#cards-ablations");
  if (elA) {
    elA.innerHTML =
      nullCard +
      calCard +
      clipCard +
      accCard;
  }

  const elC = $("#cards-clinical");
  if (elC) {
    elC.innerHTML =
      thCard +
      wdCard;
  }
}
findings().catch(err => {
  console.error(err);
  ["#cards-forensics", "#cards-ablations", "#cards-clinical"].forEach(s => {
    const el = $(s);
    if (el) el.innerHTML = `<p class="err">${err.message}</p>`;
  });
});
