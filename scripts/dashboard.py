from __future__ import annotations

import argparse
import csv
import json
import statistics
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "training_scores_dqn.csv"
DEFAULT_BASELINE = ROOT / "reports" / "random_baseline_900.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dashboard for a training run.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def read_data(input_path: Path, baseline_path: Path) -> dict:
    train: list[dict] = []
    eval_groups: dict[int, list[dict]] = {}
    if input_path.exists():
        with input_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if not row.get("episode") or not row.get("score"):
                    continue
                item = {
                    "ep": int(row["episode"]),
                    "score": float(row["score"]),
                    "rooms": int(row.get("rooms") or 0),
                    "kills": int(row.get("kills") or 0),
                    "floors": int(row.get("floors") or 0),
                    "steps": int(row.get("steps") or 0),
                    "wall": float(row.get("wall_seconds") or 0),
                    "epsilon": float(row.get("epsilon") or 0),
                    "global_step": int(row.get("global_step") or 0),
                }
                if row.get("kind") == "train":
                    train.append(item)
                elif row.get("kind") == "eval":
                    eval_groups.setdefault(item["ep"], []).append(item)

    evals = [
        {
            "ep": ep,
            "score": statistics.mean(r["score"] for r in rows),
            "rooms": sum(r["rooms"] for r in rows),
            "kills": sum(r["kills"] for r in rows),
            "floors": sum(r["floors"] for r in rows),
        }
        for ep, rows in sorted(eval_groups.items())
    ]

    baseline = None
    if baseline_path.exists():
        with baseline_path.open(newline="", encoding="utf-8") as file:
            scores = [float(r["score"]) for r in csv.DictReader(file) if r.get("score")]
        baseline = statistics.mean(scores) if scores else None

    return {"train": train, "evals": evals, "baseline": baseline}


PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Suivi entraînement DQN</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19; --page: #0d0d0d;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    }
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--page); color: var(--ink);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 20px; max-width: 1060px; margin: 0 auto;
  }
  h1 { font-size: 17px; font-weight: 650; }
  .sub { color: var(--ink-2); font-size: 12px; margin: 2px 0 14px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 14px; }
  .tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 9px 12px; }
  .tile .k { color: var(--muted); font-size: 11px; }
  .tile .v { font-size: 20px; font-weight: 650; margin-top: 1px; }
  .tile .d { color: var(--ink-2); font-size: 11px; }
  .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 14px 6px; margin-bottom: 14px; position: relative; }
  .card h2 { font-size: 13px; font-weight: 650; }
  .legend { display: flex; gap: 14px; margin: 4px 0 2px; font-size: 12px; color: var(--ink-2); flex-wrap: wrap; }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
  svg { display: block; width: 100%; height: auto; }
  .tip {
    position: absolute; pointer-events: none; display: none;
    background: var(--surface); border: 1px solid var(--ring); border-radius: 6px;
    padding: 6px 9px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    white-space: nowrap; z-index: 2;
  }
  .tip b { font-variant-numeric: tabular-nums; }
  details { margin: 10px 0 20px; }
  summary { cursor: pointer; color: var(--ink-2); font-size: 12px; }
  table { border-collapse: collapse; margin-top: 8px; font-size: 12px; font-variant-numeric: tabular-nums; }
  th, td { padding: 3px 12px 3px 0; text-align: right; }
  th { color: var(--muted); font-weight: 500; }
  td:first-child, th:first-child { text-align: left; }
</style>
</head>
<body>
<h1>Suivi de l'entraînement DQN</h1>
<div class="sub" id="status">chargement…</div>
<div class="tiles" id="tiles"></div>

<div class="card" id="card1">
  <h2>Score par épisode</h2>
  <div class="legend">
    <span style="--c:var(--s1)">entraînement (moy. mobile 15)</span>
    <span style="--c:var(--s2)">évaluation greedy</span>
    <span style="--c:var(--muted)">baseline aléatoire</span>
  </div>
  <svg id="chart1" viewBox="0 0 1020 300"></svg>
  <div class="tip" id="tip1"></div>
</div>

<div class="card" id="card2">
  <h2>Progression dans le donjon (moyenne mobile 15, entraînement)</h2>
  <div class="legend">
    <span style="--c:var(--s3)">kills / épisode</span>
    <span style="--c:var(--s4)">salles / épisode</span>
    <span style="--c:var(--ink)">▲ étage terminé</span>
  </div>
  <svg id="chart2" viewBox="0 0 1020 300"></svg>
  <div class="tip" id="tip2"></div>
</div>

<details><summary>Dernières évaluations (table)</summary><div id="evalTable"></div></details>

<script>
const W = 1020, H = 300, M = {t: 14, r: 150, b: 26, l: 44};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function movingAvg(values, w) {
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const from = Math.max(0, i - w + 1);
    let sum = 0;
    for (let j = from; j <= i; j++) sum += values[j];
    out.push(sum / (i - from + 1));
  }
  return out;
}

function scales(xMax, yMax) {
  const x = v => M.l + (v / Math.max(1, xMax)) * (W - M.l - M.r);
  const y = v => H - M.b - (v / Math.max(1e-9, yMax)) * (H - M.t - M.b);
  return {x, y};
}

function axisTicks(max, n) {
  const step = Math.max(1, Math.round(max / n));
  const ticks = [];
  for (let v = 0; v <= max; v += step) ticks.push(v);
  return ticks;
}

function line(pts, color, width) {
  if (!pts.length) return "";
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round"/>`;
}

function frame(sc, xMax, yMax, yTicks) {
  let s = "";
  for (const v of yTicks) {
    const yy = sc.y(v);
    s += `<line x1="${M.l}" y1="${yy}" x2="${W - M.r}" y2="${yy}" stroke="${css('--grid')}" stroke-width="1"/>`;
    s += `<text x="${M.l - 7}" y="${yy + 4}" text-anchor="end" font-size="11" fill="${css('--muted')}">${v}</text>`;
  }
  for (const v of axisTicks(xMax, 8)) {
    s += `<text x="${sc.x(v)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="${css('--muted')}">${v}</text>`;
  }
  s += `<line x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}" stroke="${css('--axis')}" stroke-width="1"/>`;
  return s;
}

function endLabel(x, y, text, color) {
  return `<text x="${x + 8}" y="${y + 4}" font-size="12" font-weight="600" fill="${color}">${text}</text>`;
}

let hoverData = {};

function drawChart1(data) {
  const t = data.train, svg = document.getElementById("chart1");
  if (!t.length) { svg.innerHTML = ""; return; }
  const xMax = Math.max(t[t.length - 1].ep, 20);
  const ma = movingAvg(t.map(r => r.score), 15);
  const yMax = Math.max(...ma, ...data.evals.map(e => e.score), data.baseline || 0, 20) * 1.12;
  const sc = scales(xMax, yMax);
  let s = frame(sc, xMax, yMax, axisTicks(Math.round(yMax), 5));

  if (data.baseline != null) {
    const by = sc.y(data.baseline);
    s += `<line x1="${M.l}" y1="${by}" x2="${W - M.r}" y2="${by}" stroke="${css('--muted')}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    s += endLabel(W - M.r, by, "aléatoire " + data.baseline.toFixed(1), css('--muted'));
  }
  const maPts = t.map((r, i) => [sc.x(r.ep), sc.y(ma[i])]);
  s += line(maPts, css('--s1'), 2);
  if (maPts.length) s += endLabel(...maPts[maPts.length - 1], "train " + ma[ma.length - 1].toFixed(1), css('--s1'));

  const evPts = data.evals.map(e => [sc.x(e.ep), sc.y(e.score)]);
  s += line(evPts, css('--s2'), 1.2);
  for (const [x, y] of evPts) {
    s += `<circle cx="${x}" cy="${y}" r="4" fill="${css('--s2')}" stroke="${css('--surface')}" stroke-width="2"/>`;
  }
  if (evPts.length) {
    const last = data.evals[data.evals.length - 1];
    s += endLabel(...evPts[evPts.length - 1], "éval " + last.score.toFixed(1), css('--s2'));
  }
  svg.innerHTML = s + `<g id="cross1"></g>`;
  hoverData.c1 = {sc, t, ma, evals: data.evals, xMax};
}

function drawChart2(data) {
  const t = data.train, svg = document.getElementById("chart2");
  if (!t.length) { svg.innerHTML = ""; return; }
  const xMax = Math.max(t[t.length - 1].ep, 20);
  const maK = movingAvg(t.map(r => r.kills), 15);
  const maR = movingAvg(t.map(r => r.rooms), 15);
  const yMax = Math.max(...maK, ...maR, 3) * 1.15;
  const sc = scales(xMax, yMax);
  let s = frame(sc, xMax, yMax, axisTicks(Math.ceil(yMax), 4));

  const kPts = t.map((r, i) => [sc.x(r.ep), sc.y(maK[i])]);
  const rPts = t.map((r, i) => [sc.x(r.ep), sc.y(maR[i])]);
  s += line(kPts, css('--s3'), 2);
  s += line(rPts, css('--s4'), 2);
  if (kPts.length) s += endLabel(...kPts[kPts.length - 1], "kills " + maK[maK.length - 1].toFixed(1), css('--s3'));
  if (rPts.length) s += endLabel(...rPts[rPts.length - 1], "salles " + maR[maR.length - 1].toFixed(2), css('--s4'));

  for (const r of t) {
    if (r.floors > 0) {
      const x = sc.x(r.ep);
      s += `<path d="M${x} ${M.t + 6} l5 9 h-10 z" fill="${css('--ink')}"/>`;
    }
  }
  svg.innerHTML = s + `<g id="cross2"></g>`;
  hoverData.c2 = {sc, t, maK, maR, xMax};
}

function attachHover(svgId, tipId, crossId, render) {
  const svg = document.getElementById(svgId), tip = document.getElementById(tipId);
  const card = svg.closest(".card");
  svg.addEventListener("mousemove", ev => {
    const d = render.data();
    if (!d) return;
    const rect = svg.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (W / rect.width);
    const ep = Math.round(((px - M.l) / (W - M.l - M.r)) * d.xMax);
    const i = d.t.findIndex(r => r.ep >= ep);
    if (i < 0 || px < M.l || px > W - M.r) { tip.style.display = "none"; document.getElementById(crossId).innerHTML = ""; return; }
    const row = d.t[i];
    const x = d.sc.x(row.ep);
    document.getElementById(crossId).innerHTML =
      `<line x1="${x}" y1="${M.t}" x2="${x}" y2="${H - M.b}" stroke="${css('--axis')}" stroke-width="1" stroke-dasharray="3 3"/>`;
    tip.innerHTML = render.text(row, i);
    tip.style.display = "block";
    const cardRect = card.getBoundingClientRect();
    let lx = ev.clientX - cardRect.left + 14;
    if (lx > cardRect.width - 190) lx -= 210;
    tip.style.left = lx + "px";
    tip.style.top = (ev.clientY - cardRect.top - 14) + "px";
  });
  svg.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    document.getElementById(crossId).innerHTML = "";
  });
}

attachHover("chart1", "tip1", "cross1", {
  data: () => hoverData.c1,
  text: (row, i) => {
    const d = hoverData.c1;
    const ev = d.evals.filter(e => e.ep <= row.ep).pop();
    return `épisode <b>${row.ep}</b><br>score : <b>${row.score.toFixed(2)}</b> (moy. <b>${d.ma[i].toFixed(2)}</b>)` +
      (ev ? `<br>dernière éval (ép. ${ev.ep}) : <b>${ev.score.toFixed(2)}</b>` : "");
  },
});
attachHover("chart2", "tip2", "cross2", {
  data: () => hoverData.c2,
  text: (row, i) => {
    const d = hoverData.c2;
    return `épisode <b>${row.ep}</b><br>kills : <b>${row.kills}</b> (moy. <b>${d.maK[i].toFixed(1)}</b>)` +
      `<br>salles : <b>${row.rooms}</b> (moy. <b>${d.maR[i].toFixed(2)}</b>)` +
      `<br>étages : <b>${row.floors}</b> · steps : <b>${row.steps}</b>`;
  },
});

function tile(k, v, d) {
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d || ""}</div></div>`;
}

function refresh() {
  fetch("data").then(r => r.json()).then(data => {
    const t = data.train;
    const el = document.getElementById("tiles");
    if (!t.length) {
      document.getElementById("status").textContent = "en attente de données… (" + new Date().toLocaleTimeString() + ")";
      el.innerHTML = "";
      return;
    }
    const last = t[t.length - 1];
    const recent = t.slice(-40);
    const rate = recent.reduce((a, r) => a + r.steps, 0) / Math.max(0.1, recent.reduce((a, r) => a + r.wall, 0));
    const floors = t.reduce((a, r) => a + r.floors, 0) + data.evals.reduce((a, e) => a + e.floors, 0);
    const best = data.evals.length ? Math.max(...data.evals.map(e => e.score)) : null;
    const lastEval = data.evals[data.evals.length - 1];
    el.innerHTML =
      tile("Épisode", last.ep, "epsilon " + last.epsilon.toFixed(3)) +
      tile("Steps totaux", (last.global_step / 1000).toFixed(0) + "k", rate.toFixed(0) + " steps/s") +
      tile("Record éval", best != null ? best.toFixed(2) : "—",
           data.baseline != null ? "aléatoire : " + data.baseline.toFixed(2) : "") +
      tile("Dernière éval", lastEval ? lastEval.score.toFixed(2) : "—", lastEval ? "épisode " + lastEval.ep : "") +
      tile("Étages terminés", floors, "cumul train + éval");
    document.getElementById("status").textContent =
      "mise à jour " + new Date().toLocaleTimeString() + " — rafraîchissement toutes les 5 s";
    drawChart1(data);
    drawChart2(data);
    const rows = data.evals.slice(-12).map(e =>
      `<tr><td>${e.ep}</td><td>${e.score.toFixed(2)}</td><td>${e.rooms}</td><td>${e.kills}</td><td>${e.floors}</td></tr>`).join("");
    document.getElementById("evalTable").innerHTML =
      `<table><tr><th>épisode</th><th>score moyen</th><th>salles</th><th>kills</th><th>étages</th></tr>${rows}</table>`;
  }).catch(() => {
    document.getElementById("status").textContent = "serveur injoignable — relancer scripts/dashboard.py";
  });
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    input_path: Path
    baseline_path: Path

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.rstrip("/") in ("", "/index.html"):
            body = PAGE.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif self.path.startswith("/data"):
            body = json.dumps(read_data(self.input_path, self.baseline_path)).encode("utf-8")
            content_type = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def main() -> None:
    args = parse_args()
    Handler.input_path = args.input
    Handler.baseline_path = args.baseline
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port} (Ctrl+C pour arrêter)")
    print(f"input={args.input}")
    print(f"baseline={args.baseline}")
    server.serve_forever()


if __name__ == "__main__":
    main()
