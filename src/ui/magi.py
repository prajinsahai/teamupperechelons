"""The bizmax Core panel: six agent nodes around a central hub, MAGI-style.

Rendered as self-contained HTML/CSS (SVG connectors + absolutely positioned boxes)
for `st.components.v1.html`. Node states:
  idle     - dark box, muted label
  skipped  - dimmed, router did not select it
  running  - strobes blue <-> green (the blueprint's magi-strobe, two colours)
  done     - solid green lock-in
  error    - red border
"""

from __future__ import annotations

NODES = [  # key, label, ordinal, (x%, y%) of box centre, box (w%, h%)
    ("actuary", "ACTUARY", 1, (50, 16), (30, 24)),
    ("claims", "CLAIMS ANALYST", 2, (14, 40), (26, 22)),
    ("reinsurance", "REINSURANCE MGR", 3, (86, 40), (26, 22)),
    ("capital", "CAPITAL MGR", 4, (14, 80), (26, 22)),
    ("risk", "RISK MGR", 5, (86, 80), (26, 22)),
    ("policy", "POLICY ANALYST", 6, (50, 86), (30, 22)),
]
HUB = (50, 50)

CSS = """
<style>
  .magi { position: relative; width: 100%; aspect-ratio: 16 / 8.6; background: #060a12; border: 2px solid #b45309;
          font-family: "Segoe UI", Consolas, monospace; color: #f59e0b; overflow: hidden; box-sizing: border-box; }
  .magi svg.wires { position: absolute; inset: 0; width: 100%; height: 100%; }
  .node { position: absolute; transform: translate(-50%, -50%); box-sizing: border-box; display: flex; flex-direction: column;
          align-items: center; justify-content: center; background: #38bdf8; color: #06131f; border: 3px solid #0F172A;
          font-weight: 800; letter-spacing: 0.06em; text-align: center; transition: all .3s ease; }
  .node .lbl { font-size: clamp(9px, 1.55vw, 18px); line-height: 1.1; }
  .node .sub { font-size: clamp(7px, 0.9vw, 11px); font-weight: 600; opacity: .8; margin-top: 3px; }
  .node.idle { background: #0e2a3a; color: #7aa7c2; border-color: #0F172A; }
  .node.skipped { background: #0a1a26; color: #3f5a6d; border-color: #0F172A; opacity: .55; }
  .node.running { animation: magi-strobe 1.1s infinite; }
  .node.done { background: #22c55e; color: #052e16; border-color: #00E5FF; box-shadow: 0 0 14px rgba(0,229,255,.55); }
  .node.error { background: #7f1d1d; color: #fee2e2; border-color: #ef4444; }
  @keyframes magi-strobe {
    0%   { background: #00E5FF; border-color: #00E5FF; box-shadow: 0 0 16px rgba(0,229,255,.75), inset 0 0 10px rgba(0,229,255,.25); color:#06131f; }
    50%  { background: #22c55e; border-color: #22c55e; box-shadow: 0 0 16px rgba(34,197,94,.75),  inset 0 0 10px rgba(34,197,94,.25);  color:#052e16; }
    100% { background: #00E5FF; border-color: #00E5FF; box-shadow: 0 0 16px rgba(0,229,255,.75), inset 0 0 10px rgba(0,229,255,.25); color:#06131f; }
  }
  .hub { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 22%; height: 16%;
         background: #060a12; border: 3px solid #f59e0b; display: flex; align-items: center; justify-content: center;
         font-size: clamp(14px, 2.6vw, 34px); font-weight: 900; letter-spacing: .12em; color: #f59e0b; z-index: 2; }
  .hub.busy { animation: hub-pulse 1.1s infinite; }
  @keyframes hub-pulse { 0%,100% { box-shadow: 0 0 0 rgba(245,158,11,0);} 50% { box-shadow: 0 0 22px rgba(245,158,11,.7);} }
  .corner { position: absolute; font-size: clamp(8px, 1vw, 12px); line-height: 1.25; z-index: 3; }
  .corner.tl { left: 1.2%; top: 2%; } .corner.tr { right: 1.2%; top: 2%; text-align: right; }
  .corner.bl { left: 1.2%; bottom: 2%; } .corner.br { right: 1.2%; bottom: 2%; text-align: right; }
  .corner .big { font-size: clamp(11px, 1.6vw, 20px); font-weight: 900; letter-spacing: .1em; }
  .rule { border-top: 1px solid #16a34a; margin: 2px 0; opacity: .7; }
  .tag { display: inline-block; border: 2px solid #38bdf8; color: #38bdf8; padding: 0 6px; font-weight: 800; letter-spacing: .15em; font-size: clamp(8px, 1.1vw, 13px); }
</style>
"""

PHASE_LABEL = {"idle": "STANDBY", "routing": "ROUTING", "processing": "PROCESSING", "synthesizing": "SYNTHESIZING", "complete": "RESOLVED"}


def render_magi(statuses: dict[str, str], phase: str = "idle", session_id: str = "", mode: str = "AUTO", objective: str = "") -> str:
    """Return the HTML for the panel."""
    wires = []
    for key, label, n, (x, y), (w, h) in NODES:
        col = {"running": "#00E5FF", "done": "#22c55e", "error": "#ef4444"}.get(statuses.get(key, "idle"), "#b45309")
        wires.append(f'<line x1="{HUB[0]}" y1="{HUB[1]}" x2="{x}" y2="{y}" stroke="{col}" stroke-width="1.1" vector-effect="non-scaling-stroke" opacity="0.95"/>')
    boxes = []
    for key, label, n, (x, y), (w, h) in NODES:
        st = statuses.get(key, "idle")
        sub = {"running": "ANALYSING", "done": "COMPLETE", "error": "ERROR", "skipped": "NOT ROUTED", "idle": "STANDBY"}[st]
        boxes.append(f'<div class="node {st}" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;"><div class="lbl">{label} · {n}</div><div class="sub">{sub}</div></div>')
    busy = "busy" if phase in ("routing", "processing", "synthesizing") else ""
    active = sum(1 for s in statuses.values() if s in ("running", "done"))
    obj = (objective or "").replace("<", "&lt;")[:90]
    html = f"""{CSS}
<div class="magi">
  <svg class="wires" viewBox="0 0 100 100" preserveAspectRatio="none">{''.join(wires)}</svg>
  <div class="corner tl"><div class="big">QUERY</div><div class="rule"></div>CODE : {session_id[-6:].upper() or '------'}<br>FILE : BIZMAX_CORE<br>NODES : {active}/6 ACTIVE<br>MODE : {mode}<br>PRIORITY : AAA</div>
  <div class="corner tr"><div class="big">RESOLUTION</div><div class="rule"></div><span class="tag">{PHASE_LABEL.get(phase, phase.upper())}</span></div>
  <div class="corner bl">objective : {obj or '—'}</div>
  <div class="corner br">bizmax · AI actuarial core</div>
  <div class="hub {busy}">CORE</div>
  {''.join(boxes)}
</div>"""
    return html
