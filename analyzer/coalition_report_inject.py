#!/usr/bin/env python3
"""
Inject a "Coalition Sensitivity" section into report/index.html.

Runs after coalition_sweep.py. Reads report/coalition_sweep.json and splices
a new <h2>…</h2>…<div class="box"> block into the existing report just above
the Bottom-Line verdict, plus a paragraph of prose explaining why multiplayer
GTO is a band. Idempotent — strips any previously-injected block first.
"""

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPORT = HERE / "report" / "index.html"
DATA = HERE / "report" / "coalition_sweep.json"

START = "<!-- COALITION_SECTION_START -->"
END = "<!-- COALITION_SECTION_END -->"


def build_section(data):
    sweep = data["sweep"]
    rows = data["rows"]
    policy = data.get("policy", "rule_based")
    sims = data.get("sims", "?")

    leader_deltas = np.array([r["leader_delta"] for r in rows])
    max_drifts = np.array([r["max_drift"] for r in rows])
    ld_mean = float(leader_deltas.mean())
    ld_med = float(np.median(leader_deltas))
    ld_p10 = float(np.percentile(leader_deltas, 10))
    ld_p90 = float(np.percentile(leader_deltas, 90))
    mx_mean = float(max_drifts.mean())
    mx_p90 = float(np.percentile(max_drifts, 90))
    mx_max = float(max_drifts.max())
    held = sum(1 for r in rows if np.argmax(r["all_probs"][-1]) == r["leader"])
    held_pct = 100.0 * held / len(rows) if rows else 0.0

    # Per-CP mean leader prob across positions
    leader_curve = []
    for i, cp in enumerate(sweep):
        vals = [r["leader_probs"][i] for r in rows]
        leader_curve.append(float(np.mean(vals)))

    js_sweep = json.dumps(sweep)
    js_curve = json.dumps(leader_curve)
    js_ld = json.dumps(leader_deltas.tolist())
    js_mx = json.dumps(max_drifts.tolist())

    verdict_colour = "good" if abs(ld_mean) < 0.02 else (
        "warn" if abs(ld_mean) < 0.05 else "bad")
    verdict_word = ("coalition-insensitive" if abs(ld_mean) < 0.02 else
                    ("mildly coalition-sensitive" if abs(ld_mean) < 0.05 else
                     "strongly coalition-sensitive"))

    return f"""{START}
<h2>Part E — Coalition Sensitivity: Is GTO a Point or a Band?</h2>
<div class="note">
<strong>What this shows:</strong> In a four-player general-sum game, Nash equilibrium
is <em>not unique</em> — the set of GTO strategies depends on how the non-leaders
coordinate against the leader. We sweep the engine's <code>coalition_pressure</code>
knob (0 = selfish best-response, 1 = default, 2 = strong focus-fire against the VP
leader) over {len(rows)} positions and measure how much the leader's win probability
moves. A narrow band means the single-number estimate is honest; a wide band means
the engine's output should be quoted as <code>[cp=0, cp=2]</code>, not a point.
</div>

<div class="grid">
  <div class="card"><div class="cl">Mean leader Δ (cp=2 − cp=0)</div>
    <div class="cv {verdict_colour}">{ld_mean*100:+.2f} pp</div>
    <div class="cs">median {ld_med*100:+.2f} pp · p10 {ld_p10*100:+.2f} · p90 {ld_p90*100:+.2f}</div></div>
  <div class="card"><div class="cl">Mean band width (per position)</div>
    <div class="cv">{mx_mean*100:.2f} pp</div>
    <div class="cs">p90 {mx_p90*100:.2f} · max {mx_max*100:.2f}</div></div>
  <div class="card"><div class="cl">Leader identity preserved at cp=2</div>
    <div class="cv">{held_pct:.0f}%</div>
    <div class="cs">{held} / {len(rows)} positions</div></div>
  <div class="card"><div class="cl">Verdict</div>
    <div class="cv {verdict_colour}">{verdict_word}</div>
    <div class="cs">policy {policy} · {sims} sims</div></div>
</div>

<div class="box"><div id="coalition_curve" style="height:360px"></div></div>
<div class="two">
  <div class="box"><div id="coalition_delta_hist" style="height:300px"></div></div>
  <div class="box"><div id="coalition_drift_hist" style="height:300px"></div></div>
</div>

<div class="note warn-note">
<strong>How to read this:</strong> If mean leader Δ is close to 0 (green), the
engine's single-number win probability is robust — focus-fire doesn't change who
wins, just by how much. If it's strongly negative (red), a coalition against the
leader materially changes the prediction, and the honest output for the user is
the full band, not a point estimate. The UI's <em>Coalition pressure</em> slider
exposes this knob so users can see the band themselves.
</div>

<script>
(function() {{
  const BG='#1a2e44', PL='#0d1b2a', TX='#e0e8f4', GR='#2d4a6e', MU='#7a9bbf', AC='#4a9eff';
  const base = (t,x,y) => ({{
    paper_bgcolor:BG, plot_bgcolor:PL, font:{{color:TX}},
    title:{{text:t,font:{{color:MU,size:13}}}},
    xaxis:{{title:x,gridcolor:GR,zerolinecolor:GR}},
    yaxis:{{title:y,gridcolor:GR,zerolinecolor:GR}},
    margin:{{t:50,r:30,b:50,l:60}},
    legend:{{bgcolor:'rgba(0,0,0,0)',font:{{color:TX}}}},
  }});
  const sweep = {js_sweep};
  const curve = {js_curve};
  Plotly.newPlot('coalition_curve',
    [{{x:sweep, y:curve, mode:'lines+markers', line:{{color:AC,width:3}}, marker:{{size:10}}, name:'mean leader P(win)'}}],
    base('Mean leader win-probability vs coalition pressure','coalition_pressure','P(leader wins)'),
    {{displayModeBar:false}});
  const ld = {js_ld};
  Plotly.newPlot('coalition_delta_hist',
    [{{x:ld.map(v=>v*100), type:'histogram', nbinsx:18, marker:{{color:AC}}}}],
    base('Per-position leader Δ (pp)','cp=2 − cp=0, pp','count'),
    {{displayModeBar:false}});
  const mx = {js_mx};
  Plotly.newPlot('coalition_drift_hist',
    [{{x:mx.map(v=>v*100), type:'histogram', nbinsx:18, marker:{{color:'#f39c12'}}}}],
    base('Per-position band width (pp)','max drift across cp, pp','count'),
    {{displayModeBar:false}});
}})();
</script>
{END}
"""


def main():
    if not DATA.exists():
        print(f"ERROR: {DATA} not found — run coalition_sweep.py first.")
        return
    if not REPORT.exists():
        print(f"ERROR: {REPORT} not found — run analyze.py first.")
        return

    with open(DATA) as f:
        data = json.load(f)
    if not data.get("rows"):
        print("ERROR: no rows in coalition_sweep.json.")
        return

    html = REPORT.read_text(encoding="utf-8")
    # Strip any previously-injected block (idempotent)
    html = re.sub(
        re.escape(START) + r".*?" + re.escape(END) + r"\s*",
        "",
        html,
        flags=re.DOTALL,
    )

    block = build_section(data)
    anchor = '<div class="verdict">'
    if anchor not in html:
        print(f"ERROR: could not find anchor {anchor!r} in report.")
        return
    html = html.replace(anchor, block + "\n" + anchor, 1)

    REPORT.write_text(html, encoding="utf-8")
    print(f"Injected coalition section into {REPORT}")


if __name__ == "__main__":
    main()
